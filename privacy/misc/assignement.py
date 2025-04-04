from typing import Any, List, Tuple

import numpy as np
from ortools.sat.python import cp_model
from scipy.sparse import coo_matrix


class Assignement:
    def __init__(
        self,
        patient_stay_number: List[int],
        candidate_stay_number: List[int],
        n_stays_patient: int,
    ) -> None:
        """Class to check if, given a set of precompatibles stays between patient and candidate,
        there exists a solution where all stays have one and only one assignement.
        As it is a combinatorial problem we use an optimized Constraint Programming (CP) solver.

        The parmeters `patient_stay_number` and `candidate_stay_number` are paired vectors.
        It means that for the example:
            patient_stay_number = [1, 1, 2, 4, 3]
            candidate_stay_number = [1, 3, 2, 4, 3]
        Patient's stay 1 is compatible with candidate's stay 1 and 3
        Patient's stay 2 is compatible with candidate's stay 2
        Patient's stay 4 is compatible with candidate's stay 4
        Patient's stay 3 is compatible with candidate's stay 3

        It could be represented as [(1,1),(1,3),(2,2),(4,4),(3,3)]


        Parameters
        ----------
        patient_stay_number : List[int]
            Sequence of patient's stays number that are 'pre-compatible' with candidate's stays number.
            len(patient_stay_number)==len(candidate_stay_number)
        candidate_stay_number : List[int]
            Sequence of candidate's stays number that are 'pre-compatible' with patient's stays number.
        n_stays_patient : int
            total number of stays of patient

        Examples
        --------
        patient_stay_number = [1, 1, 2, 4, 3]
        candidate_stay_number = [1, 3, 2, 4, 3]
        n = 4
        Assignement(patient_stay_number, candidate_stay_number, n)()
        >>> True

        patient_stay_number = [1, 1, 2, 4, 3, 2]
        candidate_stay_number = [1, 3, 2, 3, 3, 4]
        n = 4
        Assignement(patient_stay_number, candidate_stay_number, n)()
        >>> False
        """
        self.patient_stay_number = patient_stay_number
        self.candidate_stay_number = candidate_stay_number
        self.n_stays_patient = n_stays_patient

    @staticmethod
    def get_compatibility_matrix(
        patient_stay_number: List[int],
        candidate_stay_number: List[int],
        n_stays_patient: int,
    ) -> np.ndarray:
        """Returns a binary square matrix where x[i][j]==1 if patient stay i is pre-compatible with
        candidate stay j

        Parameters
        ----------
        patient_stay_number : List[int]
            Sequence of patient's stays number that are 'pre-compatible' with candidate's stays number.
            len(patient_stay_number)==len(candidate_stay_number)
        candidate_stay_number : List[int]
            Sequence of candidate's stays number that are 'pre-compatible' with patient's stays number.
        n_stays_patient : int
            total number of stays of patient

        Returns
        -------
        np.ndarray
        """
        row = np.array(patient_stay_number) - 1
        col = np.array(candidate_stay_number) - 1
        data = np.full_like(col, True)
        coo = coo_matrix((data, (row, col)), shape=(n_stays_patient, n_stays_patient))
        m = np.array(coo.todense())
        return m

    @staticmethod
    def get_constraint_programming_model(
        m: np.ndarray,
    ) -> Tuple[cp_model.CpModel, List[Any]]:
        """Build a Constraint Programming (CP) Model where all patient's stays have
        a candidate stay assigned and inversely. A patient stay should have exactly one candidate stay
        and inversely.

        Parameters
        ----------
        m : np.ndarray
            matrix of compatibility (binary)

        Returns
        -------
        Tuple[cp_model.CpModel, List[Any]]
        """
        num_patient_stays = m.shape[0]
        num_candidate_stays = m.shape[1]

        # Initialize model
        model = cp_model.CpModel()

        # Initialize variables
        x = []
        for i in range(num_patient_stays):
            t = []
            for j in range(num_candidate_stays):
                if m[i][j] == 1:
                    t.append(model.NewBoolVar(f"x[{i},{j}]"))
                else:
                    t.append(model.NewConstant(0))
            x.append(t)

        # Contraints
        # Each candidate stay is assigned to exactly one patient stay.
        for j in range(num_candidate_stays):
            model.AddExactlyOne(x[i][j] for i in range(num_patient_stays))

        # Each patient stay is assigned to exactly one candidate stay.
        for i in range(num_patient_stays):
            model.AddExactlyOne(x[i][j] for j in range(num_candidate_stays))

        return model, x

    @staticmethod
    def solve(model: cp_model.CpModel) -> Tuple[int, cp_model.CpSolver]:
        """Solve model.
        Return status and solver.

        Parameters
        ----------
        model : cp_model.CpModel

        Returns
        -------
        Tuple[int, cp_model.CpSolver]
        """
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        return status, solver

    @staticmethod
    def print_solution(
        solver: cp_model.CpSolver, status: int, x: List[Any], n: int
    ) -> None:
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print()
            for i in range(n):
                for j in range(n):
                    if solver.BooleanValue(x[i][j]):
                        print(
                            f"Patient stay {i + 1} assigned to candidate stay {j + 1}"
                        )
        else:
            print("No solution found.")

    def is_compatible(self) -> bool:
        """Returns True if the sequence of patient and candidate is compatible.

        Returns
        -------
        bool
        """
        m = self.get_compatibility_matrix(
            self.patient_stay_number, self.candidate_stay_number, self.n_stays_patient
        )
        model, _ = self.get_constraint_programming_model(m)
        status, _ = self.solve(model)

        return status == cp_model.OPTIMAL or status == cp_model.FEASIBLE

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Calls `is_compatible` method

        Returns
        -------
        bool
        """
        return self.is_compatible()
