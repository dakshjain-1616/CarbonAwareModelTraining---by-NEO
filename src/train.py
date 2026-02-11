import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

from scheduler import CarbonIntensityScheduler
from tracker import CarbonTracker
from utils import setup_logging, load_config

logger = logging.getLogger(__name__)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

class CarbonAwareTrainer:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        
        log_file = Path(self.config['output_dir']) / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        setup_logging(log_file)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.scheduler = CarbonIntensityScheduler(self.config.get('scheduler', {}))
        self.tracker = CarbonTracker(self.config.get('tracker', {}))
        
        self.batch_size = self.config['training']['batch_size']
        self.accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        self.epochs = self.config['training']['epochs']
        self.learning_rate = self.config['training'].get('learning_rate', 0.001)
        
        self.effective_batch_size = self.batch_size * self.accumulation_steps
        logger.info(f"Batch size: {self.batch_size}, Accumulation steps: {self.accumulation_steps}")
        logger.info(f"Effective batch size: {self.effective_batch_size}")
        
        self.training_metrics = {
            'losses': [],
            'epoch_times': [],
            'memory_allocated': [],
            'start_time': None,
            'end_time': None
        }
        
    def prepare_data(self):
        logger.info("Preparing MNIST dataset")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        data_dir = Path(self.config.get('data_dir', './data'))
        data_dir.mkdir(parents=True, exist_ok=True)
        
        train_dataset = datasets.MNIST(
            str(data_dir), 
            train=True, 
            download=True, 
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            str(data_dir), 
            train=False, 
            transform=transform
        )
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        
    def build_model(self):
        logger.info("Building model")
        self.model = SimpleCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Total parameters: {total_params:,}")
        
    def train_epoch(self, epoch: int):
        self.model.train()
        epoch_loss = 0
        epoch_start = time.time()
        
        self.optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss = loss / self.accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            epoch_loss += loss.item() * self.accumulation_steps
            
            if batch_idx % 100 == 0:
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1e9
                    logger.info(f'Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] '
                              f'Loss: {loss.item() * self.accumulation_steps:.4f} '
                              f'GPU Memory: {mem_allocated:.2f}GB')
                else:
                    logger.info(f'Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] '
                              f'Loss: {loss.item() * self.accumulation_steps:.4f}')
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(self.train_loader)
        
        self.training_metrics['losses'].append(avg_loss)
        self.training_metrics['epoch_times'].append(epoch_time)
        
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1e9
            self.training_metrics['memory_allocated'].append(mem)
            logger.info(f'Epoch {epoch} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}, Max GPU Memory: {mem:.2f}GB')
        else:
            logger.info(f'Epoch {epoch} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}')
            
    def evaluate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(self.test_loader)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        
        logger.info(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
        return test_loss, accuracy
        
    def train(self):
        logger.info("="*60)
        logger.info("STARTING CARBON-AWARE TRAINING PIPELINE")
        logger.info("="*60)
        
        schedule_decision = {}
        if self.config.get('scheduler', {}).get('enabled', True):
            if self.config.get('scheduler', {}).get('wait_for_low_carbon', False):
                logger.info("Waiting for low carbon intensity window...")
                schedule_decision = self.scheduler.wait_for_low_carbon()
            else:
                can_start, intensity, reason = self.scheduler.should_start_training()
                schedule_decision = {
                    'started': can_start,
                    'wait_time_seconds': 0,
                    'checks_performed': 1,
                    'final_intensity': intensity,
                    'reason': reason
                }
        else:
            logger.info("Scheduler disabled, starting immediately")
            schedule_decision = {
                'started': True,
                'wait_time_seconds': 0,
                'checks_performed': 0,
                'final_intensity': None,
                'reason': 'Scheduler disabled'
            }
        
        self.prepare_data()
        self.build_model()
        
        self.tracker.start()
        self.training_metrics['start_time'] = datetime.now().isoformat()
        
        logger.info(f"Training for {self.epochs} epochs...")
        for epoch in range(1, self.epochs + 1):
            self.train_epoch(epoch)
            
        test_loss, accuracy = self.evaluate()
        
        self.training_metrics['end_time'] = datetime.now().isoformat()
        emissions_data = self.tracker.stop()
        
        model_path = Path(self.config['output_dir']) / self.config['training']['model_save_path']
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        summary = {
            'config': self.config['training'],
            'schedule_decision': schedule_decision,
            'emissions': emissions_data,
            'training_metrics': {
                'final_loss': self.training_metrics['losses'][-1] if self.training_metrics['losses'] else None,
                'test_loss': test_loss,
                'test_accuracy': accuracy,
                'total_epochs': self.epochs,
                'start_time': self.training_metrics['start_time'],
                'end_time': self.training_metrics['end_time'],
                'avg_epoch_time': sum(self.training_metrics['epoch_times']) / len(self.training_metrics['epoch_times']) if self.training_metrics['epoch_times'] else 0,
                'max_memory_gb': max(self.training_metrics['memory_allocated']) if self.training_metrics['memory_allocated'] else 0,
                'device': str(self.device)
            }
        }
        
        summary_path = Path(self.config['output_dir']) / f"summary_{self.config['training']['run_name']}.json"
        self.tracker.save_summary(summary_path, summary)
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETED")
        logger.info("="*60)
        
        return summary

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python train.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    trainer = CarbonAwareTrainer(config_path)
    trainer.train()