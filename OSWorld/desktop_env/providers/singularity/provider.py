import logging
import os
import platform
import signal
import shutil
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
        self.process_log_path = None
        self.process_log_file = None
        self.environment = {"DISK_SIZE": "32G", "RAM_SIZE": "4G", "CPU_CORES": "4"}

        temp_dir = Path(os.getenv('TEMP') if platform.system() == 'Windows' else '/tmp')
        self.lock_file = temp_dir / "singularity_port_allocation.lck"
        self.port_registry_dir = temp_dir / "singularity_port_registry"
        self.port_registry_dir.mkdir(parents=True, exist_ok=True)
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

        # Priority: 1. Environment variable 2. SIF file (better for old kernels) 3. Directory Sandbox
        self.sandbox_path = os.getenv("OSWORLD_SANDBOX")
        if not self.sandbox_path:
            sif_path = "/public/home/xlwang/genalyu/3SPO/osworld_uitars.sif"
            dir_path = "/public/home/xlwang/genalyu/3SPO/osworld-sandbox"
            self.sandbox_path = sif_path if os.path.exists(sif_path) else dir_path

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
                error_msg = self._read_process_log_tail()
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
            logger.error(f"Container startup log tail: {self._read_process_log_tail()}")
        
        raise TimeoutError(f"VM on port {self.server_port} failed to become ready within {timeout}s")

    def _read_process_log_tail(self, max_lines: int = 80) -> str:
        if not self.process_log_path:
            return "No process log path available"
        log_path = Path(self.process_log_path)
        if not log_path.exists():
            return f"No process log file found at {log_path}"
        try:
            with open(log_path, "rb") as f:
                return b"".join(f.readlines()[-max_lines:]).decode(errors="replace").strip() or "Process log is empty"
        except Exception as e:
            return f"Failed to read process log: {e}"

    def _run_preflight(self, cmd, env, timeout=45):
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return process.returncode, stdout, stderr
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            try:
                stdout, stderr = process.communicate(timeout=3)
            except Exception:
                stdout, stderr = "", ""
            return None, stdout, stderr

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

                source_sandbox = Path(self.sandbox_path)
                runtime_root = self.local_sandbox_root / f"osworld_runtime_{os.getpid()}"
                if runtime_root.exists():
                    shutil.rmtree(runtime_root, ignore_errors=True)
                runtime_root.mkdir(parents=True, exist_ok=True)

                # Create runtime directories for mounting
                runtime_run_dir = runtime_root / "run"
                runtime_run_dir.mkdir(parents=True, exist_ok=True)
                # Don't create shm as directory, script wants to symlink it
                # (runtime_run_dir / "shm").mkdir(parents=True, exist_ok=True) 

                # Create local tmp and xdg runtime for the container
                runtime_tmp_dir = runtime_root / "tmp"
                runtime_tmp_dir.mkdir(parents=True, exist_ok=True)
                runtime_xdg_dir = runtime_root / "xdg"
                runtime_xdg_dir.mkdir(parents=True, exist_ok=True)

                # Create storage directory for QEMU
                runtime_storage_dir = runtime_root / "storage"
                runtime_storage_dir.mkdir(parents=True, exist_ok=True)

                # Make everything in runtime_root world-writable to avoid permission issues inside container
                subprocess.run(f"chmod -R 777 {runtime_root}", shell=True)

                # Create nginx directories
                runtime_nginx_lib = runtime_root / "var_lib_nginx"
                runtime_nginx_log = runtime_root / "var_log_nginx"
                runtime_misc_dir = runtime_root / "var_lib_misc"
                runtime_nginx_path = runtime_root / "nginx"
                runtime_nginx_lib.mkdir(parents=True, exist_ok=True)
                runtime_nginx_log.mkdir(parents=True, exist_ok=True)
                runtime_misc_dir.mkdir(parents=True, exist_ok=True)
                runtime_nginx_path.mkdir(parents=True, exist_ok=True)

                # If the source is a directory, copy existing /run content to avoid shadowing entry.sh
                dir_path = Path("/public/home/xlwang/genalyu/3SPO/osworld-sandbox")
                source_run_dir = dir_path / "run"
                source_nginx_path = dir_path / "etc/nginx"

                # Priority copy for run directory
                if source_run_dir.exists() and source_run_dir.is_dir():
                    for item in os.listdir(source_run_dir):
                        s = source_run_dir / item
                        d = runtime_run_dir / item
                        if s.is_dir():
                            if d.exists():
                                shutil.rmtree(d)
                            shutil.copytree(s, d, symlinks=True)
                        else:
                            shutil.copy2(s, d)
                elif source_sandbox.is_dir() and (source_sandbox / "run").exists():
                    source_run_dir = source_sandbox / "run"
                    for item in os.listdir(source_run_dir):
                        s = source_run_dir / item
                        d = runtime_run_dir / item
                        if s.is_dir():
                            if d.exists():
                                shutil.rmtree(d)
                            shutil.copytree(s, d, symlinks=True)
                        else:
                            shutil.copy2(s, d)
                
                # Patch ALL files in runtime_run_dir recursively to redirect /boot.qcow2
                # This is necessary for read-only SIF images where / is not writable
                logger.info(f"Patching all scripts in {runtime_run_dir} to redirect /boot.qcow2 to /storage/boot.qcow2...")
                subprocess.run(f"find {runtime_run_dir} -type f | xargs sed -i 's|/boot.qcow2|/storage/boot.qcow2|g' 2>/dev/null || true", shell=True)
                
                # Hard patch KVM check in any script that might be doing it
                subprocess.run(f"find {runtime_run_dir} -type f | xargs sed -i 's|if \[ ! -w /dev/kvm \]|if false|g' 2>/dev/null || true", shell=True)
                subprocess.run(f"find {runtime_run_dir} -type f | xargs sed -i 's|\[ ! -w /dev/kvm \]|false|g' 2>/dev/null || true", shell=True)
                # Also patch the shell-based write test in proc.sh
                subprocess.run(f"find {runtime_run_dir} -type f | xargs sed -i \"s|if ! sh -c 'echo -n > /dev/kvm' &> /dev/null; then|if false; then|g\" 2>/dev/null || true", shell=True)
                # Ensure it doesn't exit even if it thinks KVM is missing
                subprocess.run(f"find {runtime_run_dir} -type f | xargs sed -i 's|exit 88|true|g' 2>/dev/null || true", shell=True)
                # Force KVM_ERR to stay empty
                subprocess.run(f"find {runtime_run_dir} -type f | xargs sed -i 's|KVM_ERR=\"(device file missing)\"|KVM_ERR=\"\"|g' 2>/dev/null || true", shell=True)
                subprocess.run(f"find {runtime_run_dir} -type f | xargs sed -i 's|KVM_ERR=\"(no write access)\"|KVM_ERR=\"\"|g' 2>/dev/null || true", shell=True)
                subprocess.run(f"find {runtime_run_dir} -type f | xargs sed -i 's|KVM_ERR=\"(vmx/svm disabled)\"|KVM_ERR=\"\"|g' 2>/dev/null || true", shell=True)
                # Bypassing the entire KVM_ERR error block
                subprocess.run(f"find {runtime_run_dir} -type f | xargs sed -i 's|if \[ -n \"\$KVM_ERR\" \]; then|if false; then|g' 2>/dev/null || true", shell=True)
                
                # Patch out root privilege checks (since we run as non-root in Singularity)
                logger.info("Patching out root privilege checks in scripts...")
                subprocess.run(f"find {runtime_run_dir} -type f | xargs sed -i 's|\\[ \"$(id -u)\" -ne \"0\" \\]|false|g' 2>/dev/null || true", shell=True)
                
                # Also patch /run/shm to a writable path just in case
                subprocess.run(f"find {runtime_run_dir} -type f | xargs sed -i 's|/run/shm|/tmp/shm_fake|g' 2>/dev/null || true", shell=True)
                (runtime_tmp_dir / "shm_fake").mkdir(parents=True, exist_ok=True, mode=0o777)

                # Create a fake 'id' command to bypass root checks inside container
                fake_id_path = runtime_root / "fake_id"
                with open(fake_id_path, "w") as f:
                    f.write("#!/bin/sh\necho 0\n")
                os.chmod(fake_id_path, 0o755)

                # Priority copy for nginx config
                if source_nginx_path.exists() and source_nginx_path.is_dir():
                    shutil.copytree(source_nginx_path, runtime_nginx_path, symlinks=True, dirs_exist_ok=True)
                elif source_sandbox.is_dir() and (source_sandbox / "etc/nginx").exists():
                    shutil.copytree(source_sandbox / "etc/nginx", runtime_nginx_path, symlinks=True, dirs_exist_ok=True)

                if any(runtime_nginx_path.iterdir()):
                    logger.info(f"Modifying local nginx config to use ports (API: {self.server_port}, VNC: {self.vnc_port})...")
                    # Replace port 80 (standard API) with our dynamic server_port
                    subprocess.run(f"find {runtime_nginx_path} -type f | xargs sed -i 's/listen 80;/listen {self.server_port};/g' 2>/dev/null || true", shell=True)
                    subprocess.run(f"find {runtime_nginx_path} -type f | xargs sed -i 's/listen \\[::\\]:80;/listen \\[::\\]:{self.server_port};/g' 2>/dev/null || true", shell=True)
                    # Replace port 8006 (standard VNC) with our dynamic vnc_port
                    subprocess.run(f"find {runtime_nginx_path} -type f | xargs sed -i 's/8006/{self.vnc_port}/g' 2>/dev/null || true", shell=True)
                    
                    nginx_conf = runtime_nginx_path / "nginx.conf"
                    if nginx_conf.exists():
                        # Disable nginx 'user' directive since we run as non-root
                        subprocess.run(f"sed -i 's/^user /#user /g' {nginx_conf}", shell=True)
                        # Fix nginx pid and lock file locations to be writable
                        subprocess.run(f"sed -i 's|/run/nginx.pid|/tmp/nginx.pid|g' {nginx_conf}", shell=True)
                else:
                    logger.warning("Could not find source nginx config to patch! Nginx may fail to bind port 80.")
                    runtime_nginx_path = None

                # KVM acceleration is critical
                kvm_flag = []
                kvm_env = "Y"
                if os.path.exists("/dev/kvm") and os.access("/dev/kvm", os.W_OK):
                    # We trust the host's 666 permission (as shown in diag)
                    # and bind it directly. We set KVM=Y to force container to use it.
                    kvm_flag = ["--bind", "/dev/kvm:/dev/kvm"]
                else:
                    error_msg = (
                        "\n" + "="*80 + "\n"
                        "CRITICAL ERROR: KVM acceleration is required but not accessible!\n"
                        "The Singularity container cannot start the VM with KVM acceleration because:\n"
                        "1. /dev/kvm does not exist OR\n"
                        "2. Current user " + os.getlogin() + " (ID: " + str(os.getuid()) + ") lacks write permission to /dev/kvm.\n\n"
                        "TO FIX THIS, please ask your administrator to:\n"
                        "   Option A: Run 'sudo chmod 666 /dev/kvm' on the compute node.\n"
                        "   Option B: Add you to the 'kvm' group: 'sudo usermod -aG kvm " + os.getlogin() + "'\n"
                        "="*80 + "\n"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

                # Clean up host environment variables that might interfere with container binaries
                # Especially LD_LIBRARY_PATH and PYTHONPATH on cluster environments
                env = os.environ.copy()
                for var in ["LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME", "PERL5LIB"]:
                    if var in env:
                        del env[var]
                
                env.update(self.environment)
                singularity_tmp = runtime_root / "singularity_tmp"
                singularity_cache = runtime_root / "singularity_cache"
                singularity_tmp.mkdir(parents=True, exist_ok=True)
                singularity_cache.mkdir(parents=True, exist_ok=True)
                env["SINGULARITY_TMPDIR"] = str(singularity_tmp)
                env["SINGULARITY_CACHEDIR"] = str(singularity_cache)
                env["SINGULARITY_DISABLE_CACHE"] = "True"
                # Singularity uses SINGULARITYENV_ prefix to pass vars into the container
                env.update({
                    "SINGULARITYENV_VNC_PORT": str(self.vnc_port),
                    "SINGULARITYENV_SERVER_PORT": str(self.server_port),
                    "SINGULARITYENV_CHROMIUM_PORT": str(self.chromium_port),
                    "SINGULARITYENV_VLC_PORT": str(self.vlc_port),
                    "SINGULARITYENV_VM_NET_DEV": "lo", # Fix 'eth0 not found' error
                    "SINGULARITYENV_KVM": kvm_env, # Bypass KVM check if not available
                    "SINGULARITYENV_DHCP": "N", # Bypass bridge creation/DHCP inside container
                    "SINGULARITYENV_NETWORK": "user", # Force usermode networking
                    "SINGULARITYENV_KVM_FORCE": "Y", # Additional flag for some qemu-docker versions
                    "SINGULARITYENV_XDG_RUNTIME_DIR": "/xdg", # Local writable XDG path
                    "VNC_PORT": str(self.vnc_port),
                    "SERVER_PORT": str(self.server_port),
                    "CHROMIUM_PORT": str(self.chromium_port),
                    "VLC_PORT": str(self.vlc_port),
                    "USER": "root", # Fake being root for internal scripts
                    "HOME": "/root" # Container scripts often expect /root
                })

                # Define the entry script path. Usually /run/entry.sh or /entry.sh
                # We'll check which one exists in the source sandbox or directory
                entry_script = "/run/entry.sh"
                dir_path = Path("/public/home/xlwang/genalyu/3SPO/osworld-sandbox")
                if not (runtime_run_dir / "entry.sh").exists() and \
                   not (source_sandbox.is_dir() and (source_sandbox / "run/entry.sh").exists()) and \
                   not (dir_path.is_dir() and (dir_path / "run/entry.sh").exists()):
                    if (source_sandbox.is_dir() and (source_sandbox / "entry.sh").exists()) or \
                       (dir_path.is_dir() and (dir_path / "entry.sh").exists()):
                        entry_script = "/entry.sh"
                    else:
                        # Fallback to run if we can't find entry script
                        entry_script = None

                preflight_modes = [
                    ["--cleanenv", "--no-home", "--dev", "--writable-tmpfs", "--no-mount", "overlay"],
                    ["--cleanenv", "--no-home", "--dev", "--writable-tmpfs"],
                    ["--cleanenv", "--no-home", "--dev", "--no-mount", "overlay"],
                    ["--cleanenv", "--no-home", "--dev"],
                    ["--cleanenv", "--containall", "--dev"],
                    ["--cleanenv", "--dev"],
                    ["--dev"],
                    []
                ]
                selected_mode = None
                preflight_failures = []
                for mode_flags in preflight_modes:
                    # Check if KVM is accessible inside this mode
                    preflight_inner_cmd = "echo preflight_ok && ls -l /dev/kvm && [ -w /dev/kvm ] && echo kvm_writable || echo kvm_not_writable"
                    preflight_cmd = [
                        "singularity", "exec",
                        *mode_flags,
                        "--bind", f"{runtime_run_dir}:/run",
                        "--bind", f"{runtime_run_dir}:/var/run",
                        "--bind", f"{runtime_storage_dir}:/storage",
                        "--bind", f"{runtime_tmp_dir}:/tmp",
                        "--bind", f"{runtime_xdg_dir}:/xdg",
                        "--bind", f"{runtime_nginx_lib}:/var/lib/nginx",
                        "--bind", f"{runtime_nginx_log}:/var/log/nginx",
                        "--bind", f"{runtime_misc_dir}:/var/lib/misc",
                        *kvm_flag,
                        str(source_sandbox),
                        "/bin/sh", "-c", preflight_inner_cmd
                    ]
                    return_code, stdout, stderr = self._run_preflight(preflight_cmd, env, timeout=45)
                    
                    mode_str = ' '.join(mode_flags) if mode_flags else '(none)'
                    if return_code == 0 and "preflight_ok" in stdout:
                        kvm_status = "WRITABLE" if "kvm_writable" in stdout else "NOT WRITABLE"
                        logger.info(f"Preflight mode '{mode_str}' worked. KVM inside: {kvm_status}")
                        if "kvm_writable" in stdout:
                            selected_mode = mode_flags
                            break
                        # If KVM not writable, keep searching but remember this as a fallback if nothing better works
                        if selected_mode is None:
                            selected_mode = mode_flags
                    else:
                        err = f"mode='{mode_str}' failed: rc={return_code}, stderr={stderr[-200:] if stderr else 'empty'}"
                        preflight_failures.append(err)
                        logger.warning(err)

                if selected_mode is None:
                    raise RuntimeError(f"Singularity preflight failed: {' | '.join(preflight_failures)}")

                cmd = [
                    "singularity",
                    "exec" if entry_script else "run",
                    *selected_mode,
                    "--bind", f"{fake_id_path}:/usr/bin/id",
                    "--bind", f"{fake_id_path}:/bin/id",
                    *kvm_flag,
                    "--bind", f"{os.path.abspath(path_to_vm)}:/System.qcow2",
                ]

                if runtime_nginx_path:
                    cmd.extend(["--bind", f"{runtime_nginx_path}:/etc/nginx"])

                cmd.extend([
                    "--bind", f"{runtime_run_dir}:/run",
                    "--bind", f"{runtime_run_dir}:/var/run",
                    "--bind", f"{runtime_storage_dir}:/storage",
                    "--bind", f"{runtime_tmp_dir}:/tmp",
                    "--bind", f"{runtime_xdg_dir}:/xdg",
                    "--bind", f"{runtime_nginx_lib}:/var/lib/nginx",
                    "--bind", f"{runtime_nginx_log}:/var/log/nginx",
                    "--bind", f"{runtime_misc_dir}:/var/lib/misc"
                ])
                cmd.append(str(source_sandbox))
                
                if entry_script:
                    cmd.extend(["/bin/bash", entry_script])

                logger.info(f"Starting Singularity (Port {self.server_port}, mode: {' '.join(selected_mode) if selected_mode else '(none)'}): {' '.join(cmd)}")
                self.process_log_path = str(runtime_root / "singularity_startup.log")
                self.process_log_file = open(self.process_log_path, "ab")
                self.process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=self.process_log_file,
                    stderr=self.process_log_file,
                    preexec_fn=os.setsid 
                )

                # Store for cleanup
                self.fake_id_path = fake_id_path
                self.runtime_root = runtime_root

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
                # Use psutil to find all child processes (like QEMU) and kill them recursively
                parent = psutil.Process(self.process.pid)
                children = parent.children(recursive=True)
                for child in children:
                    try:
                        child.kill()
                    except:
                        pass
                parent.kill()
                self.process.wait(timeout=2)
            except (psutil.NoSuchProcess, ProcessLookupError, OSError):
                pass
            finally:
                self.process = None
        if self.process_log_file:
            try:
                self.process_log_file.close()
            except:
                pass
            self.process_log_file = None
            self.process_log_path = None
        
        # Cleanup regardless of process state
        self._release_ports()
        if hasattr(self, 'fake_id_path') and self.fake_id_path.exists():
            try:
                self.fake_id_path.unlink()
            except:
                pass
        if hasattr(self, 'runtime_root') and self.runtime_root.exists():
            try:
                shutil.rmtree(self.runtime_root)
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
