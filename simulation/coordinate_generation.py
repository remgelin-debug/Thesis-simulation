"""Optional independently seeded coordinates for each configured network site."""
import json
import math
import random

from .models import Record
from .return_generation import derive_stream_seed

COORDINATE_SITE_GROUPS = (('collection_point', 'collection_points'),
                          ('plant', 'plants'), ('recycler', 'recycling_facilities'))


def validate_coordinate_generation(settings: Record) -> None:
    """Require ordered finite axis bounds and finite spans, allowing a fixed coordinate on either axis."""
    if not isinstance(settings, dict):
        raise ValueError('coordinate_generation must define x_range and y_range bounds')
    for name in ('x_range', 'y_range'):
        bounds = settings.get(name)
        if (not isinstance(bounds, list) or len(bounds) != 2 or
                any(type(value) not in (int, float) or not math.isfinite(value) for value in bounds) or
                bounds[0] > bounds[1] or not math.isfinite(bounds[1] - bounds[0])):
            raise ValueError(f'Coordinate {name} must contain ordered finite bounds with a finite span')


def derive_coordinate_seed(master_seed: int, site_type: str, site_id: str) -> int:
    """Identify one site's geometry stream by master seed, type and ID, independent of all other generators."""
    encoded = json.dumps(['coordinates-v1', site_type, site_id], separators=(',', ':'), ensure_ascii=True)
    return derive_stream_seed(master_seed, encoded)


def generate_site_coordinates(scenario: Record, master_seed: int) -> Record:
    """Fill a copied scenario's site coordinates with two uniform draws per private stream; return seed records."""
    settings = scenario['coordinate_generation']
    validate_coordinate_generation(settings)
    seeds = {}
    for site_type, group in COORDINATE_SITE_GROUPS:
        seeds[site_type] = {}
        for site in scenario[group]:
            seed = derive_coordinate_seed(master_seed, site_type, site['id'])
            seeds[site_type][site['id']] = seed
            generator = random.Random(seed)
            site['coordinates'] = [generator.uniform(*settings['x_range']), generator.uniform(*settings['y_range'])]
    return seeds
