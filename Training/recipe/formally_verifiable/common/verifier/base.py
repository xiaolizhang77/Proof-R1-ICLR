from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseVerifier(ABC):


    @abstractmethod
    def verify_step(
        self,
        dependencies: List[str],
        conclusion: str,
        premises_fol: Dict[str, str],
        all_steps_fol: Dict[str, str],
    ) -> Dict[str, Any]:


        pass


    @abstractmethod
    def batch_verify(
        self,
        steps: List[Dict[str, Any]],
        premises_fol: Dict[str, str],
    ) -> List[Dict[str, Any]]:


        pass
