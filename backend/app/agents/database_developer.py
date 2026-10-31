from backend.app.agents.developer import DeveloperAgent


class DatabaseDeveloper(DeveloperAgent):

    def __init__(self):
        super().__init__(specialization="database")