from dgllife.data import MUV, BACE, BBBP, ClinTox, SIDER, ToxCast, HIV, PCBA, Tox21
from dgllife.utils import smiles_to_bigraph
from functools import partial

def load_dataset(args):
    if args['dataset'] == 'MUV':
        from dgllife.data import MUV
        dataset = MUV(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                        node_featurizer=args['node_featurizer'],
                        edge_featurizer=args['edge_featurizer'],
                        n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
    elif args['dataset'] == 'BACE':
        from dgllife.data import BACE
        dataset = BACE(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                        node_featurizer=args['node_featurizer'],
                        edge_featurizer=args['edge_featurizer'],
                        n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
    elif args['dataset'] == 'BBBP':
        from dgllife.data import BBBP
        dataset = BBBP(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                        node_featurizer=args['node_featurizer'],
                        edge_featurizer=args['edge_featurizer'],
                        n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
    elif args['dataset'] == 'ClinTox':
        from dgllife.data import ClinTox
        dataset = ClinTox(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                            node_featurizer=args['node_featurizer'],
                            edge_featurizer=args['edge_featurizer'],
                            n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
    elif args['dataset'] == 'SIDER':
        from dgllife.data import SIDER
        dataset = SIDER(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                        node_featurizer=args['node_featurizer'],
                        edge_featurizer=args['edge_featurizer'],
                        n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
    elif args['dataset'] == 'ToxCast':
        from dgllife.data import ToxCast
        dataset = ToxCast(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                            node_featurizer=args['node_featurizer'],
                            edge_featurizer=args['edge_featurizer'],
                            n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
    elif args['dataset'] == 'HIV':
        from dgllife.data import HIV
        dataset = HIV(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                        node_featurizer=args['node_featurizer'],
                        edge_featurizer=args['edge_featurizer'],
                        n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
    elif args['dataset'] == 'PCBA':
        from dgllife.data import PCBA
        dataset = PCBA(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                        node_featurizer=args['node_featurizer'],
                        edge_featurizer=args['edge_featurizer'],
                        n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
    elif args['dataset'] == 'Tox21':
        from dgllife.data import Tox21
        dataset = Tox21(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                        node_featurizer=args['node_featurizer'],
                        edge_featurizer=args['edge_featurizer'],
                        n_jobs=1 if args['num_workers'] == 0 else args['num_workers'])
    return dataset