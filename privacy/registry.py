import os

from confit import Registry, set_default_registry

user = os.environ.get("USER")


@set_default_registry
class PrivacyRegistry:
    cohort_generator = Registry(
        (f"{user}_registry", "cohort_generator"), entry_points=True
    )
    pseudonymizer = Registry((f"{user}_registry", "pseudonymizer"), entry_points=True)
    indicators = Registry((f"{user}_registry", "indicators"), entry_points=True)
    plots = Registry((f"{user}_registry", "plots"), entry_points=True)

    _catalogue = dict(
        cohort_generator=cohort_generator,
        pseudonymizer=pseudonymizer,
        indicators=indicators,
        plots=plots,
    )


registry = PrivacyRegistry
