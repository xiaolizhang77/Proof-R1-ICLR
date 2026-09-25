from typing import List, Dict, Any
from .base import BaseVerifier
from ..fol_converter import FOLToZ3Converter, verify_implication


class Z3Verifier(BaseVerifier):


    def __init__(self, timeout_ms: int = 5000):
        self.converter = FOLToZ3Converter()
        self.timeout_ms = timeout_ms


    def verify_step(
        self,
        dependencies: List[str],
        conclusion: str,
        premises_fol: Dict[str, str],
        all_steps_fol: Dict[str, str],
    ) -> Dict[str, Any]:


        premises_z3 = []
        missing = []
        for dep in dependencies:
            if dep in premises_fol:
                fol_str = premises_fol[dep]
            elif dep in all_steps_fol:
                fol_str = all_steps_fol[dep]
            else:
                missing.append(dep)
                continue
            try:
                z3_expr = self.converter.convert(fol_str)
                premises_z3.append(z3_expr)
            except Exception as e:
                return {
                    "verified": False,
                    "error": f"Failed to parse dependency '{dep}': {e}",
                    "details": {"dependency": dep, "fol": fol_str},
                }

        if missing:
            return {
                "verified": False,
                "error": f"Missing dependencies: {missing}",
                "details": {"missing": missing},
            }


        try:
            conclusion_z3 = self.converter.convert(conclusion)
        except Exception as e:
            return {
                "verified": False,
                "error": f"Failed to parse conclusion: {e}",
                "details": {"conclusion": conclusion},
            }


        try:
            verified, msg = verify_implication(premises_z3, conclusion_z3, self.timeout_ms)
        except Exception as e:
            return {
                "verified": False,
                "error": f"Failed to verify implication: {e}",
                "details": {"conclusion": conclusion},
            }
        return {
            "verified": verified,
            "error": None if verified else msg,
            "details": {
                "premises_count": len(premises_z3),
                "message": msg,
            },
        }


    def batch_verify(
        self,
        steps: List[Dict[str, Any]],
        premises_fol: Dict[str, str],
    ) -> List[Dict[str, Any]]:

        all_steps_fol: Dict[str, str] = {}
        results = []
        for step in steps:
            res = self.verify_step(
                dependencies=step.get("dependencies", []),
                conclusion=step.get("conclusion", ""),
                premises_fol=premises_fol,
                all_steps_fol=all_steps_fol,
            )
            res["step_id"] = step.get("id")
            results.append(res)

            if res["verified"]:
                all_steps_fol[step.get("id", "")] = step.get("conclusion", "")
        return results
