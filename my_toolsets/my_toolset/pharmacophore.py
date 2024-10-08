from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D.SigFactory import SigFactory

fdef = """
DefineFeature Hydrophobic [$([$([#6&!H0]);!$([#6][$([#7,#8,#15,$([#6,#16]=[O,N])])])]),$([$([#16&D2]);!$([#16][$([#7,#8,#15])])]),Cl,Br,I]
  Family LH
  Weights 1.0
EndFeature
DefineFeature Donor [$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]
  Family HD
  Weights 1.0
EndFeature
DefineFeature Acceptor [$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N&v3;H1,H2]-[!$(*=[O,N,P,S])]),$([N;v3;H0]),$([n,o,s;+0]),F]
  Family HA
  Weights 1.0
EndFeature
DefineFeature BasicGroup [$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))]),$([N,n;X2;+0])]
  Family BG
  Weights 1.0
EndFeature
DefineFeature AcidicGroup [$([C,S](=[O,S,P])-[O;H1])]
  Family AG
  Weights 1.0
EndFeature

"""
defaultBins = [(2, 3), (3, 4), (4, 6), (5, 8), (7, 10), (9, 13), (11, 16), (14, 21)]

def get_factory():
  featFactory = ChemicalFeatures.BuildFeatureFactoryFromString(fdef)
  factory = SigFactory(featFactory, minPointCount=2, maxPointCount=3, useCounts=True, trianglePruneBins=False)
  factory.SetBins(defaultBins)
  factory.Init()
  return factory

factory = get_factory()

if (__name__ == "__main__"):
  print("Number of bins: {}".format(factory.GetNumBins()))
  print("Number of features: {}".format(factory.GetSigSize()))
  print(1+2)
