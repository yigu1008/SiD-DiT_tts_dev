from typing import TYPE_CHECKING

from diffusers.utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_flax_available,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}
_additional_imports = {}
_import_structure = {"pipeline_output": ["SiDPipelineOutput"]}

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from diffusers.utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["pipeline_sid_sd3"] = ["SiDSD3Pipeline"]
    _import_structure["pipeline_sid_flux"] = ["SiDFluxPipeline"]
    _import_structure["pipeline_sid_sana"] = ["SiDSanaPipeline"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from diffusers.utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .pipeline_sid_sd3 import SiDSD3Pipeline
        from .pipeline_sid_flux import SiDFluxPipeline
        from .pipeline_sid_sana import SiDSanaPipeline
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
    for name, value in _additional_imports.items():
        setattr(sys.modules[__name__], name, value)
