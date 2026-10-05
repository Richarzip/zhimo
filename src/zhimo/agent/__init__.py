def create_calligraphy_agent(*args, **kwargs):
    from .builder import create_calligraphy_agent as _create_calligraphy_agent
    return _create_calligraphy_agent(*args, **kwargs)

__all__ = ["create_calligraphy_agent"]
