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
        self.port_registry_dir = temp_dir / "singularity_port_registry"
        self.port_registry_dir.mkdir(parents=True, exist_ok=True)
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

        # Default Sandbox path, can be overridden by environment variable
        self.sandbox_path = os.getenv("OSWORLD_SANDBOX", "/public/home/xlwang/genalyu/3SPO/osworld-sandbox")
        # Local cache path in /tmp to avoid NFS latency and permission issues
        self.local_sandbox_root = Path("/tmp/osworld_cache")
        self.local_sandbox_root.mkdir(parents=True, exist_ok=True)

    def _get_used_ports(self):
        """Get all currently used ports and reserved ports."""
        system_ports = set(conn.laddr.port for conn in psutil.net_connections())
        # Also check our internal registry for ports reserved by other processes
        reserved_ports = set()
        for p_file in self.port_registry_dir.glob("port_*"):
            try:
                reserved_ports.add(int(p_file.name.split("_")[1]))
            except:
                pass
        return system_ports | reserved_ports

    def _reserve_port(self, port):
        """Mark a port as reserved."""
        (self.port_registry_dir / f"port_{port}").touch()

    def _release_ports(self):
        """Release all ports reserved by this instance."""
        for port in [self.vnc_port, self.server_port, self.chromium_port, self.vlc_port]:
            if port:
                p_file = self.port_registry_dir / f"port_{port}"
                if p_file.exists():
                    try:
                        p_file.unlink()
                    except:
                        pass

    def _get_available_port(self, start_port: int) -> int:
        """Find next available port and reserve it."""
        used_ports = self._get_used_ports()
        port = start_port
        while port < 65354:
            if port not in used_ports:
                self._reserve_port(port)
                return port
            port += 1
        raise PortAllocationError(f"No available ports found starting from {start_port}")

    def _wait_for_vm_ready(self, timeout: int = 1400):
        """Wait for VM to be ready by checking screenshot endpoint."""
        # Use longer timeout for software emulation mode (no KVM)
        start_time = time.time()
        
        def check_screenshot():
            # Check if the process is still alive
            if self.process and self.process.poll() is not None:
                # Process has exited, read error output
                _, stderr = self.process.communicate()
                error_msg = stderr.decode() if stderr else "No error message"
                logger.error(f"Singularity process died. Error: {error_msg}")
                raise RuntimeError(f"Singularity process died: {error_msg}")

            try:
                response = requests.get(
                    f"http://localhost:{self.server_port}/screenshot",
                    timeout=(5, 5)
                )
                return response.status_code == 200
            except Exception:
                return False

        while time.time() - start_time < timeout:
            if check_screenshot():
                return True
            time.sleep(RETRY_INTERVAL)
        
        if self.process:
            logger.error(f"Timeout reached for port {self.server_port}. Checking process status...")
        
        raise TimeoutError(f"VM on port {self.server_port} failed to become ready within {timeout}s")

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str, name=None):
        # Use a single lock for all port allocation and container startup
        lock = FileLock(str(self.lock_file))
        
        try:
            with lock:
                # Add jitter to avoid simultaneous port scanning
                import random
                time.sleep(random.uniform(0, 3))

                # Allocate ports
                self.vnc_port = self._get_available_port(8006 + (os.getpid() % 100))
                self.server_port = self._get_available_port(5000 + (os.getpid() % 100))
                self.chromium_port = self._get_available_port(9222 + (os.getpid() % 100))
                self.vlc_port = self._get_available_port(8080 + (os.getpid() % 100))

                if not os.path.exists(self.sandbox_path):
                    raise FileNotFoundError(f"Sandbox directory not found: {self.sandbox_path}")

                # Create a local copy of the sandbox in /tmp for performance and isolation
                # We use a unique directory for each process to avoid interference
                local_sandbox = self.local_sandbox_root / f"osworld_sandbox_{os.getpid()}"
                
                # Check if sandbox is valid (has basic dir structure and copy is complete)
                is_valid_sandbox = local_sandbox.exists() and (local_sandbox / ".copy_complete").exists()

                if not is_valid_sandbox:
                    if local_sandbox.exists():
                        logger.info(f"Removing invalid sandbox at {local_sandbox}...")
                        import shutil
                        shutil.rmtree(local_sandbox, ignore_errors=True)
                    
                    logger.info(f"Creating local sandbox copy at {local_sandbox} (this may take a few minutes)...")
                    # Use 'rsync' instead of 'cp -a' to be more robust with symlinks and partial copies
                    temp_copy_path = f"{local_sandbox}.tmp"
                    subprocess.run(f"rm -rf {temp_copy_path} && mkdir -p {temp_copy_path} && rsync -a {self.sandbox_path}/ {temp_copy_path}/", shell=True, check=True)
                    (Path(temp_copy_path) / ".copy_complete").touch()
                    os.rename(temp_copy_path, local_sandbox)
                
                # IMPORTANT: Singularity with --writable requires destination mount points to exist in the sandbox.
                # Common cluster mount points and standard system ones:
                for mount_point in ["public", "tmp", "dev", "proc", "sys", "storage", "gpfs", "var/lib/nginx", "run/nginx"]:
                    (local_sandbox / mount_point).mkdir(parents=True, exist_ok=True)
                
                # Create a fake 'id' command to bypass root checks inside container
                fake_id_path = self.local_sandbox_root / f"fake_id_{os.getpid()}"
                with open(fake_id_path, "w") as f:
                    f.write("#!/bin/sh\necho 0\n")
                os.chmod(fake_id_path, 0o755)

                # Patch nginx config in the LOCAL sandbox copy
                nginx_config_path = local_sandbox / "etc/nginx"
                if nginx_config_path.exists():
                    logger.info(f"Modifying local nginx config to use ports (API: {self.server_port}, VNC: {self.vnc_port})...")
                    # Replace port 80 (standard API) with our dynamic server_port
                    subprocess.run(f"find {nginx_config_path} -type f | xargs sed -i 's/listen 80;/listen {self.server_port};/g' 2>/dev/null || true", shell=True)
                    subprocess.run(f"find {nginx_config_path} -type f | xargs sed -i 's/listen \\[::\\]:80;/listen \\[::\\]:{self.server_port};/g' 2>/dev/null || true", shell=True)
                    # Replace port 8006 (standard VNC) with our dynamic vnc_port
                    subprocess.run(f"find {nginx_config_path} -type f | xargs sed -i 's/8006/{self.vnc_port}/g' 2>/dev/null || true", shell=True)
                    
                    nginx_conf = nginx_config_path / "nginx.conf"
                    if nginx_conf.exists():
                        # Disable nginx 'user' directive since we run as non-root
                        subprocess.run(f"sed -i 's/^user /#user /g' {nginx_conf}", shell=True)
                        # Fix nginx pid and lock file locations to be writable
                        subprocess.run(f"sed -i 's|/run/nginx.pid|/tmp/nginx.pid|g' {nginx_conf}", shell=True)

                # KVM acceleration is critical
                kvm_flag = []
                if os.path.exists("/dev/kvm"):
                    # Check if current user can access /dev/kvm
                    if os.access("/dev/kvm", os.R_OK | os.W_OK):
                        kvm_flag = ["--bind", "/dev/kvm:/dev/kvm"]
                    else:
                        logger.warning("KVM exists but current user lacks permission! VM will be slow.")
                else:
                    logger.warning("KVM not found! Using software emulation (slow).")

                # Clean up host environment variables that might interfere with container binaries
                # Especially LD_LIBRARY_PATH and PYTHONPATH on cluster environments
                env = os.environ.copy()
                for var in ["LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME", "PERL5LIB"]:
                    if var in env:
                        del env[var]
                
                env.update(self.environment)
                # Singularity uses SINGULARITYENV_ prefix to pass vars into the container
                env.update({
                    "SINGULARITYENV_VNC_PORT": str(self.vnc_port),
                    "SINGULARITYENV_SERVER_PORT": str(self.server_port),
                    "SINGULARITYENV_CHROMIUM_PORT": str(self.chromium_port),
                    "SINGULARITYENV_VLC_PORT": str(self.vlc_port),
                    "VNC_PORT": str(self.vnc_port),
                    "SERVER_PORT": str(self.server_port),
                    "CHROMIUM_PORT": str(self.chromium_port),
                    "VLC_PORT": str(self.vlc_port),
                    "USER": "root", # Fake being root for internal scripts
                    "HOME": "/root" # Container scripts often expect /root
                })

                # Define the entry script path. Usually /run/entry.sh or /entry.sh
                # We'll check which one exists in the local sandbox
                entry_script = "/run/entry.sh"
                if not (local_sandbox / "run/entry.sh").exists():
                    if (local_sandbox / "entry.sh").exists():
                        entry_script = "/entry.sh"
                    else:
                        # Fallback to run if we can't find entry script
                        entry_script = None

                cmd = [
                    "singularity", "exec" if entry_script else "run",
                    # Remove --contain for now to see if it's the cause of basic binary failure
                    # "--contain", 
                    "--cleanenv", # Prevent host environment variables from interfering
                    "--no-home",  # Don't mount host home directory
                    "--writable", # Use the local sandbox copy with write permissions
                    "--bind", f"{fake_id_path}:/usr/bin/id",
                    "--bind", f"{fake_id_path}:/bin/id",
                    *kvm_flag,
                    "--bind", f"{os.path.abspath(path_to_vm)}:/System.qcow2",
                    str(local_sandbox)
                ]
                
                if entry_script:
                    cmd.extend(["/bin/bash", entry_script])

                logger.info(f"Starting Singularity (Port {self.server_port}): {' '.join(cmd)}")
                self.process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid 
                )

                # Store for cleanup
                self.fake_id_path = fake_id_path
                self.local_sandbox = local_sandbox

            # Wait for VM to be ready
            self._wait_for_vm_ready()

            logger.info(f"Started Singularity container with ports - VNC: {self.vnc_port}, "
                       f"Server: {self.server_port}, Chrome: {self.chromium_port}, VLC: {self.vlc_port}")

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
            logger.info(f"Stopping Singularity container (PID: {self.process.pid})...")
            try:
                import signal
                # Get the process group ID
                pgid = os.getpgid(self.process.pid)
                # Force kill the entire process group immediately in software mode
                # to avoid lingering QEMU processes
                os.killpg(pgid, signal.SIGKILL)
                self.process.wait(timeout=1)
            except (ProcessLookupError, OSError):
                pass
            finally:
                self.process = None
        
        # Cleanup regardless of process state
        self._release_ports()
        if hasattr(self, 'fake_id_path') and self.fake_id_path.exists():
            try:
                self.fake_id_path.unlink()
            except:
                pass
        if hasattr(self, 'local_sandbox') and self.local_sandbox.exists():
            try:
                import shutil
                shutil.rmtree(self.local_sandbox)
            except:
                pass

        self.server_port = None
        self.vnc_port = None
        self.chromium_port = None
        self.vlc_port = None
    
    def pause_emulator(self):
        # Singularity doesn't have a direct pause command like Docker
        pass

    def unpause_emulator(self):
        pass
