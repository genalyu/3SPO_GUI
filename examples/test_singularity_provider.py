import sys
import os
import logging

# Add OSWorld to sys.path to import the provider
sys.path.append(os.path.join(os.getcwd(), "OSWorld"))

from desktop_env.providers.singularity.provider import ApptainerProvider

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_provider")

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_apptainer_provider.py <path_to_vm>")
        sys.exit(1)
    
    path_to_vm = os.path.abspath(sys.argv[1])
    
    provider = ApptainerProvider()
    try:
        logger.info(f"Attempting to start emulator with VM: {path_to_vm}")
        # This will call the actual logic in provider.py, including preflight and mounts
        provider.start_emulator(path_to_vm, headless=True, os_type="Ubuntu")
        
        logger.info("SUCCESS: Emulator started and reached ready state (screenshot endpoint works)!")
        
        ip_ports = provider.get_ip_address(path_to_vm)
        logger.info(f"Provider IP/Ports: {ip_ports}")
        
    except Exception as e:
        logger.error(f"FAILURE: Emulator failed to start: {e}")
        sys.exit(1)
    finally:
        logger.info("Cleaning up...")
        provider.stop_emulator(path_to_vm)

if __name__ == "__main__":
    main()
