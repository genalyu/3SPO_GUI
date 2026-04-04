from desktop_env.providers.base import VMManager
import logging

logger = logging.getLogger("desktopenv.providers.apptainer.ApptainerVMManager")
logger.setLevel(logging.INFO)

class ApptainerVMManager(VMManager):
    def __init__(self, registry_path=""):
        pass

    def initialize_registry(self, *args, **kwargs):
        pass

    def add_vm(self, vm_path, *args, **kwargs):
        pass

    def delete_vm(self, vm_path, *args, **kwargs):
        pass

    def occupy_vm(self, vm_path, *args, **kwargs):
        pass

    def list_free_vms(self, *args, **kwargs):
        return []

    def check_and_clean(self, *args, **kwargs):
        pass

    def get_vm_path(self, *args, **kwargs):
        # For Apptainer, we usually pass the path directly via command line
        # This is a fallback if no path is provided
        return ""
