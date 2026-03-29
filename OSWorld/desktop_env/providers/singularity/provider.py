import logging
import os
import platform
import time
import psutil
import requests
from filelock import FileLock
from pathlib import Path
import subprocess
from desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.singularity.SingularityProvider")
logger.setLevel(logging.INFO)

WAIT_TIME = 3
RETRY_INTERVAL = 1
LOCK_TIMEOUT = 300


class PortAllocationError(Exception):
    pass


class SingularityProvider(Provider):
    def __init__(self, region: str = None):
        super().__init__(region)
        # Check if singularity is available
        try:
            result = subprocess.run(["singularity", "--version"], capture_output=True, check=True)
            logger.info(f"Using Singularity: {result.stdout.decode().strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Singularity not found! Please install Singularity to use this provider.")

        self.server_port = None
        self.vnc_port = None
        self.chromium_port = None
        self.vlc_port = None
        self.process = None
        self.environment = {"DISK_SIZE": "32G", "RAM_SIZE": "4G", "CPU_CORES": "4"}

        temp_dir = Path(os.getenv('TEMP') if platform.system() == 'Windows' else '/tmp')
        self.lock_file = temp_dir / "singularity_port_allocation.lck"
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

        # Default SIF image path, can be overridden by environment variable
        self.sif_image = os.getenv("OSWORLD_SIF_IMAGE", "osworld-docker.sif")

    def _get_used_ports(self):
        """Get all currently used ports on the system."""
        # Singularity uses host network by default, so we only need to check system ports
        return set(conn.laddr.port for conn in psutil.net_connections())

    def _get_available_port(self, start_port: int) -> int:
        """Find next available port starting from start_port."""
        used_ports = self._get_used_ports()
        port = start_port
        while port < 65354:
            if port not in used_ports:
                return port
            port += 1
        raise PortAllocationError(f"No available ports found starting from {start_port}")

    def _wait_for_vm_ready(self, timeout: int = 300):
        """Wait for VM to be ready by checking screenshot endpoint."""
        start_time = time.time()
        
        def check_screenshot():
            try:
                response = requests.get(
                    f"http://localhost:{self.server_port}/screenshot",
                    timeout=(10, 10)
                )
                return response.status_code == 200
            except Exception:
                return False

        while time.time() - start_time < timeout:
            if check_screenshot():
                return True
            logger.info("Checking if virtual machine is ready...")
            time.sleep(RETRY_INTERVAL)
        
        raise TimeoutError("VM failed to become ready within timeout period")

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str, name=None):
        # Use a single lock for all port allocation and container startup
        lock = FileLock(str(self.lock_file))
        
        try:
            with lock:
                # Allocate all required ports
                self.vnc_port = self._get_available_port(8006)
                self.server_port = self._get_available_port(5000)
                self.chromium_port = self._get_available_port(9222)
                self.vlc_port = self._get_available_port(8080)

                if not os.path.exists(self.sif_image):
                    raise FileNotFoundError(f"Singularity image not found at {self.sif_image}. "
                                            f"Please set OSWORLD_SIF_IMAGE environment variable or "
                                            f"place the SIF file in the current directory.")

                # Singularity run command
                # We bind the VM path to /System.qcow2 as expected by the container
                # We pass the allocated ports via environment variables
                # Note: This assumes the container's entrypoint script respects these environment variables
                env = os.environ.copy()
                env.update(self.environment)
                env.update({
                    "VNC_PORT": str(self.vnc_port),
                    "SERVER_PORT": str(self.server_port),
                    "CHROMIUM_PORT": str(self.chromium_port),
                    "VLC_PORT": str(self.vlc_port)
                })

                cmd = [
                    "singularity", "run",
                    "--nv", # Use GPU if available
                    "--bind", f"{os.path.abspath(path_to_vm)}:/System.qcow2",
                    self.sif_image
                ]

                logger.info(f"Starting Singularity container: {' '.join(cmd)}")
                self.process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid # Create a new process group to kill it properly
                )

            logger.info(f"Started Singularity container with ports - VNC: {self.vnc_port}, "
                       f"Server: {self.server_port}, Chrome: {self.chromium_port}, VLC: {self.vlc_port}")

            # Wait for VM to be ready
            self._wait_for_vm_ready()

        except Exception as e:
            logger.error(f"Error starting Singularity container: {e}")
            self.stop_emulator(path_to_vm)
            raise e

    def get_ip_address(self, path_to_vm: str) -> str:
        if not all([self.server_port, self.chromium_port, self.vnc_port, self.vlc_port]):
            raise RuntimeError("VM not started - ports not allocated")
        return f"localhost:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        raise NotImplementedError("Snapshots not available for Singularity provider")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        self.stop_emulator(path_to_vm)

    def stop_emulator(self, path_to_vm: str):
        if self.process:
            logger.info("Stopping Singularity container...")
            try:
                import signal
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=WAIT_TIME)
            except Exception as e:
                logger.error(f"Error stopping Singularity process: {e}")
                if self.process:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            finally:
                self.process = None
                self.server_port = None
                self.vnc_port = None
                self.chromium_port = None
                self.vlc_port = None
    
    def pause_emulator(self):
        # Singularity doesn't have a direct pause command like Docker
        pass

    def unpause_emulator(self):
        pass
