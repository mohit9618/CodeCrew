from backend.app.agents.developer import DeveloperAgent


class BackendDeveloper(DeveloperAgent):

    def __init__(self):
        super().__init__(specialization="backend")