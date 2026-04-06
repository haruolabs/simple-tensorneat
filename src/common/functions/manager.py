class FunctionManager:
    def __init__(self, name2jnp):
        self.name2jnp = dict(name2jnp)
        for name, func in self.name2jnp.items():
            setattr(self, name, func)

    def get_all_funcs(self):
        return [getattr(self, name) for name in self.name2jnp]

    def add_func(self, name, func):
        if not callable(func):
            raise ValueError("The provided function is not callable")
        if name in self.name2jnp:
            raise ValueError(f"The provided name={name} is already in use")
        self.name2jnp[name] = func
        setattr(self, name, func)

