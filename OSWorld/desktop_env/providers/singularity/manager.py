from desktop_env.providers.base import VMManager
import logging

logger = logging.getLogger("desktopenv.providers.singularity.SingularityVMManager")
logger.setLevel(logging.INFO)

class SingularityVMManager(VMManager):
    def __init__(self, registry_path=""):
        pass

    def add_vm(self, vm_path):
        pass

    def check_and_clean(self):
        pass

    def delete_vm(self, vm_path):
        pass

    def initialize_registry(self):
        pass
