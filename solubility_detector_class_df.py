import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator


class MolecularFeatures:

    def __init__(self, csv_path):
        """
        Load dataset and prepare RDKit molecules.
        """

        self.csv_path = csv_path

        self.df = pd.read_csv(
            csv_path,
            usecols=[
                "ESOL predicted log solubility in mols per litre",
                "smiles"
            ]
        )

        self.df.rename(
            columns={
                "ESOL predicted log solubility in mols per litre": "solubility",
                "smiles": "SMILES"
            },
            inplace=True
        )

        # SMILES → RDKit molecule
        self.df["mol"] = self.df["SMILES"].apply(Chem.MolFromSmiles)

        # Create Morgan fingerprint generator once
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048,
            includeChirality=True
        )


    def descriptor_features(self, mol):
        """
        Calculate molecular descriptors for one molecule.
        """

        return {
            "MolWt": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "NumHDonors": Descriptors.NumHDonors(mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
            "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
            "RingCount": rdMolDescriptors.CalcNumRings(mol),
            "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol)
        }


    def get_descriptors(self):
        """
        Generate descriptor DataFrame.
        """

        results = []

        for mol in self.df["mol"]:
            results.append(self.descriptor_features(mol))

        descriptor_df = pd.DataFrame(results)

        self.df = pd.concat(
            [self.df, descriptor_df],
            axis=1
        )

        return descriptor_df


    def get_fingerprints(self):
        """
        Generate Morgan fingerprints.
        """

        results = []

        for mol in self.df["mol"]:
            fingerprint = self.fpgen.GetFingerprint(mol)
            results.append(fingerprint)

        # Convert fingerprints into normal 0/1 columns
        fingerprint_df = pd.DataFrame(
            [list(fp) for fp in results]
        )

        fingerprint_df.columns = [
            f"FP_{i}" for i in range(fingerprint_df.shape[1])
        ]

        return fingerprint_df


    def get_full_dataset(self):
        """
        Return dataset containing descriptors + Morgan fingerprints.
        """

        # Add descriptors
        self.get_descriptors()

        # Generate fingerprints
        fingerprint_df = self.get_fingerprints()

        # Add fingerprints
        self.df = pd.concat(
            [self.df, fingerprint_df],
            axis=1
        )

        return self.df