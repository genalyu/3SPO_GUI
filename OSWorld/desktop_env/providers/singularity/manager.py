from desktop_env.providers.base import VMManager
import logging

logger = logging.getLogger("desktopenv.providers.singularity.SingularityVMManager")
logger.setLevel(logging.INFO)

class SingularityVMManager(VMManager):
    def __init__(self, registry_path=""):
        pass

    def initialize_registry(self, **kwargs):
        pass

    def add_vm(self, vm_path, **kwargs):
        pass

    def delete_vm(self, vm_path, **kwargs):
        pass

    def occupy_vm(self, vm_path, pid, **kwargs):
        pass

    def list_free_vms(self, **kwargs):
        return []

    def check_and_clean(self, **kwargs):
        pass

    def get_vm_path(self, **kwargs):
        # For Singularity, we usually pass the path directly via command line
        # This is a fallback if no path is provided
        return ""
