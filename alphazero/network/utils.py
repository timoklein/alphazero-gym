from typing import Any, Callable, NamedTuple, Type, Union
import torch.distributions as D
import torch.nn as nn


class NonlinearityMapping(NamedTuple):
    relu: Callable[..., nn.Module] = nn.ReLU
    leakyrelu: Callable[..., nn.Module] = nn.LeakyReLU
    relu6: Callable[..., nn.Module] = nn.ReLU6
    swish: Callable[..., nn.Module] = nn.SiLU
    silu: Callable[..., nn.Module] = nn.SiLU
    elu: Callable[..., nn.Module] = nn.ELU
    hardswish: Callable[..., nn.Module] = nn.Hardswish


def _map_call_dict(call_dict: NonlinearityMapping, element: Any,) -> Any:
    if isinstance(element, str):
        element = _process_str(element)
        return getattr(call_dict, element)
    else:
        return element


def _map_nonlinearities(
    element: Any, nonlinearity_mapping: Type[NonlinearityMapping] = NonlinearityMapping
) -> Any:
    """Checks whether a string input specifies a PyTorch layer.  
        
        The method checks if the input is a string.  
        If the input is a string, it is preprocessed and then mapped to
        a corresponding PyTorch activation layer.  
        If the input is not a string it is returned unchanged.  

        Parameters  
        ----------  
        element : Any  
            Arbitrary input to this function.  
        
        Returns  
        -------  
        Any  
            Returns either a callable activation or normalization layer 
            or the input element.   
        """

    nonlinearities = nonlinearity_mapping()

    return _map_call_dict(nonlinearities, element)


def _process_str(string: str) -> str:
    """Lower case, strip trailing whitespace and replace _ in a string.  
    
    Parameters  
    ----------  
    string : str  
        Input string.  
    
    Returns  
    -------  
    str  
        Processed string.  
    """
    return string.lower().strip().replace("_", "")
