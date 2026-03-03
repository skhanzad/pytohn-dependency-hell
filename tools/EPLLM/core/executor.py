# Minimal executor for EPLLM (naughty_bois, update_llm_eval, shuffle_modules).
# Standalone; no dependency on pllm TestExecutor.
from .py_pi_query import PyPIQuery


class Executor:
    def __init__(self, base_modules="./modules", logging=False) -> None:
        self.pypi = PyPIQuery(logging=logging, base_modules=base_modules)

    def naughty_bois(self, module, error_handler, error_type, llm_eval):
        error_handler[error_type] = error_handler.get(error_type, 0) + 1
        error_handler['previous'] = error_type
        if module is not None and isinstance(module, dict) and module.get('module') in llm_eval.get('python_modules', {}):
            mod_name = module['module']
            if mod_name in error_handler.get('error_modules', {}):
                error_handler['error_modules'][mod_name].append(llm_eval['python_modules'][mod_name])
            else:
                error_handler['error_modules'] = error_handler.get('error_modules', {})
                error_handler['error_modules'][mod_name] = [llm_eval['python_modules'][mod_name]]
        else:
            if error_handler.get('error_modules') is None:
                error_handler['error_modules'] = {}
        return error_handler

    def update_llm_eval(self, new, llm_eval):
        details = llm_eval.copy()
        details['previous_python_modules'] = details.get('python_modules', {}).copy()
        if isinstance(details['previous_python_modules'], list):
            details['previous_python_modules'] = {}
        if new is not None:
            module_name = self.pypi.check_module_name(new.get('module', ''))
            module_name = module_name[0] if len(module_name) > 0 else module_name
            ver = new.get('version')
            if ver in (None, 'None', 'none', '') and module_name in details.get('python_modules', {}):
                details['python_modules'].pop(module_name, None)
            else:
                details['python_modules'][module_name] = ver
        return details

    @staticmethod
    def append_module(module_name, lst):
        return module_name in lst

    def shuffle_modules(self, new_module, move_module, llm_details):
        modules = []
        python_modules = llm_details.get('python_modules', {}).copy()
        for module in python_modules:
            if module == move_module:
                if not self.append_module(new_module, modules):
                    modules.append(new_module)
                if not self.append_module(move_module, modules):
                    modules.append(move_module)
            else:
                if not self.append_module(module, modules):
                    modules.append(module)
        llm_details['python_modules'] = {m: python_modules[m] for m in modules}
        return llm_details
