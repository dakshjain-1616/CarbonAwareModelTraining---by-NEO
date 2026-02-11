import logging
import json
from pathlib import Path
from codecarbon import EmissionsTracker
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class CarbonTracker:
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config.get('output_dir', './output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.project_name = config.get('project_name', 'carbon-aware-training')
        self.measure_power_secs = config.get('measure_power_secs', 15)
        
        self.tracker: Optional[EmissionsTracker] = None
        self.emissions_data: Dict = {}
        
    def start(self):
        logger.info("Starting carbon emissions tracking")
        self.tracker = EmissionsTracker(
            project_name=self.project_name,
            output_dir=str(self.output_dir),
            measure_power_secs=self.measure_power_secs,
            save_to_file=True,
            log_level='warning'
        )
        self.tracker.start()
        
    def stop(self) -> Dict:
        if self.tracker:
            logger.info("Stopping carbon emissions tracking")
            emissions = self.tracker.stop()
            
            self.emissions_data = {
                'emissions_kg_co2': emissions if emissions else 0,
                'project_name': self.project_name,
                'output_dir': str(self.output_dir)
            }
            
            logger.info(f"Total emissions: {self.emissions_data['emissions_kg_co2']:.6f} kg CO2")
            return self.emissions_data
        return {}
    
    def get_emissions(self) -> Dict:
        return self.emissions_data
    
    def save_summary(self, filepath: Path, additional_data: Dict = None):
        summary = {
            **self.emissions_data,
            **(additional_data or {})
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved emissions summary to {filepath}")