class orca_gym.robosuite.rror(Exception):
    """Base class for exceptions in orca_gym.robosuite."""

    pass


class XMLError(robosuiteError):
    """Exception raised for errors related to xml."""

    pass


class SimulationError(robosuiteError):
    """Exception raised for errors during runtime."""

    pass


class RandomizationError(robosuiteError):
    """Exception raised for really really bad RNG."""

    pass
