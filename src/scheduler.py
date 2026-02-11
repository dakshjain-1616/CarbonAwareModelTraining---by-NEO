import requests
import logging
import time
from datetime import datetime
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class CarbonIntensityScheduler:
    def __init__(self, config: Dict):
        self.config = config
        self.threshold = config.get('carbon_threshold', 300)
        self.max_wait_seconds = config.get('max_wait_seconds', 3600)
        self.check_interval = config.get('check_interval', 300)
        self.use_mock = config.get('use_mock_data', True)
        self.location = config.get('location', 'US')
        self.api_url = config.get('api_url', None)
        
    def get_carbon_intensity(self) -> Optional[float]:
        if self.use_mock:
            return self._get_mock_intensity()
        
        if not self.api_url:
            logger.warning("No API URL provided, falling back to mock data")
            return self._get_mock_intensity()
        
        try:
            response = requests.get(self.api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                intensity = data.get('carbonIntensity', data.get('value', None))
                if intensity:
                    logger.info(f"Real carbon intensity: {intensity} gCO2/kWh")
                    return float(intensity)
            else:
                logger.warning(f"API returned status {response.status_code}, using mock")
                return self._get_mock_intensity()
        except Exception as e:
            logger.warning(f"API call failed: {e}, using mock data")
            return self._get_mock_intensity()
    
    def _get_mock_intensity(self) -> float:
        hour = datetime.now().hour
        if 2 <= hour <= 6:
            intensity = 200 + (hour * 10)
        elif 7 <= hour <= 10:
            intensity = 350 + (hour * 5)
        elif 11 <= hour <= 17:
            intensity = 400 + ((hour - 11) * 5)
        elif 18 <= hour <= 22:
            intensity = 380 - ((hour - 18) * 10)
        else:
            intensity = 280 + (hour * 5)
        
        logger.info(f"Mock carbon intensity: {intensity} gCO2/kWh (hour={hour})")
        return intensity
    
    def should_start_training(self) -> tuple[bool, float, str]:
        intensity = self.get_carbon_intensity()
        
        if intensity is None:
            logger.warning("Could not fetch carbon intensity, proceeding anyway")
            return True, 0, "Unable to fetch carbon intensity"
        
        if intensity <= self.threshold:
            logger.info(f"Carbon intensity {intensity} <= threshold {self.threshold}, starting training")
            return True, intensity, "Low carbon intensity"
        else:
            logger.info(f"Carbon intensity {intensity} > threshold {self.threshold}, delaying")
            return False, intensity, f"High carbon intensity ({intensity} gCO2/kWh)"
    
    def wait_for_low_carbon(self) -> Dict:
        start_time = time.time()
        checks = 0
        
        logger.info(f"Waiting for carbon intensity <= {self.threshold} gCO2/kWh")
        
        while True:
            checks += 1
            can_start, intensity, reason = self.should_start_training()
            
            if can_start:
                wait_time = time.time() - start_time
                logger.info(f"Training cleared to start after {wait_time:.1f}s and {checks} checks")
                return {
                    'started': True,
                    'wait_time_seconds': wait_time,
                    'checks_performed': checks,
                    'final_intensity': intensity,
                    'reason': reason
                }
            
            elapsed = time.time() - start_time
            if elapsed >= self.max_wait_seconds:
                logger.warning(f"Max wait time {self.max_wait_seconds}s exceeded, starting anyway")
                return {
                    'started': True,
                    'wait_time_seconds': elapsed,
                    'checks_performed': checks,
                    'final_intensity': intensity,
                    'reason': 'Max wait time exceeded'
                }
            
            logger.info(f"Check {checks}: Intensity={intensity}, waiting {self.check_interval}s...")
            time.sleep(self.check_interval)