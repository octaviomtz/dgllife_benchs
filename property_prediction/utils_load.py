from dgllife.data import MUV, BACE, BBBP, ClinTox, SIDER, ToxCast, HIV, PCBA, Tox21
from dgllife.utils import smiles_to_bigraph
from functools import partial
import json
import os

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

def get_configure2(model, featurizer_type, dataset, path):
    """Query for configuration
    Parameters
    ----------
    model : str
        Model type
    featurizer_type : str
        The featurization performed
    dataset : str
        Dataset for modeling
    Returns
    -------
    dict
        Returns the manually specified configuration
    """
    if featurizer_type == 'pre_train':
        with open('{}/configures/{}/{}.json'.format(path, dataset, model), 'r') as f:
            config = json.load(f)
    else:
        file_path = '{}/configures/{}/{}_{}.json'.format(path, dataset, model, featurizer_type)
        if not os.path.isfile(file_path):
            return NotImplementedError('Model {} on dataset {} with featurization {} has not been '
                                       'supported'.format(model, dataset, featurizer_type))
        with open(file_path, 'r') as f:
            config = json.load(f)
    return config