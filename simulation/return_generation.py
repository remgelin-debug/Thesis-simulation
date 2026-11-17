"""Resolve seeded Poisson arrivals, positive batch sizes and independent product qualities."""
import copy
import hashlib
import json
import math
from pathlib import Path
import random
from .models import Record
from .inventory_config import validate_inventory_scenario


def derive_stream_seed(seed: int, stream_name: str) -> int:
    """Derive a stable stream seed from the integer seed and a named stream using SHA-256."""
    encoded = f'thesis-return-v1|{seed}|{stream_name}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(encoded).digest(), 'big')


def weighted_index(generator: random.Random, weights: list[float]) -> int:
    """Draw one index by cumulative weight using exactly one draw from the supplied stream."""
    target = generator.random() * sum(weights)
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if target < cumulative:
            return index
    return max(index for index, weight in enumerate(weights) if weight > 0)


def validate_generation_config(configuration: Record) -> None:
    """Reject invalid generation settings and positive-weight incompatible quality classes."""
    if configuration['format_version'] != 2 or 'products' in configuration or 'arrival_batches' in configuration:
        raise ValueError('Use an unresolved format-two configuration, without products or arrival_batches')
    generation = configuration['generation']
    rate, start, end = (generation[key] for key in ('arrival_rate', 'start_time', 'end_time'))
    if not all(type(value) in (int, float) and math.isfinite(value) for value in (rate, start, end)):
        raise ValueError('Arrival settings must be finite numbers')
    if rate < 0 or not 0 <= start <= end <= configuration['horizon']:
        raise ValueError('Require nonnegative rate and 0 <= start <= end <= horizon')
    batch = generation['batch_size']
    if type(batch) is int:
        if batch <= 0:
            raise ValueError('Fixed batch size must be positive')
    else:
        if not isinstance(batch, dict) or not batch['values'] or len(batch['values']) != len(batch['weights']):
            raise ValueError('Batch values and weights must be nonempty equal-length lists')
        if any(type(size) is not int or size <= 0 for size in batch['values']):
            raise ValueError('Batch sizes must be positive integers')
        validate_weights(batch['weights'])
    qualities = configuration['quality_classes']
    quality_ids = [quality['id'] for quality in qualities]
    if len(set(quality_ids)) != len(quality_ids) or not all(isinstance(identifier, str) and identifier for identifier in quality_ids):
        raise ValueError('Quality class IDs must be unique nonempty strings')
    overrides = configuration['collection_point'].get('quality_weights', {})
    if set(overrides) - set(quality_ids):
        raise ValueError('Collection-point quality weights reference unknown classes')
    operations = {station['operation_id'] for station in configuration['plant']['workstations']}
    for quality in qualities:
        validate_weights([quality['weight']], require_positive=False)
        weight = overrides.get(quality['id'], quality['weight'])
        validate_weights([weight], require_positive=False)
        if quality['disposition'] == 'recycle':
            if quality['operations']:
                raise ValueError('Recycling classes must have no remanufacturing operations')
        elif quality['disposition'] == 'remanufacture':
            if not quality['operations'] or (weight > 0 and any(operation not in operations for operation in quality['operations'])):
                raise ValueError('Positive-weight remanufacturing routes must be supported by the plant')
        else:
            raise ValueError('Unknown quality disposition')
    effective = [overrides.get(q['id'], q['weight']) for q in qualities
                 if configuration['recycling_facilities'] or q['disposition'] == 'remanufacture']
    validate_weights(effective, require_positive=rate > 0 and end > start)
    physical = copy.deepcopy(configuration)
    physical['products'] = []
    physical['seed_provenance'] = {'source': 'generation_validation'}
    validate_inventory_scenario(physical)


def validate_weights(weights: list[float], require_positive: bool = True) -> None:
    """Check finite nonnegative weights, a finite total and optional positive effective mass."""
    if not all(type(weight) in (int, float) and math.isfinite(weight) and weight >= 0 for weight in weights):
        raise ValueError('Weights must be finite and nonnegative')
    total = sum(weights)
    if not math.isfinite(total) or (require_positive and total <= 0):
        raise ValueError('Sampling requires positive finite effective weight')


def generate_return_batches(configuration: Record, generator: random.Random) -> list[Record]:
    """Generate the complete arrival/batch stream without consulting any product qualities."""
    settings = configuration['generation']
    time = settings['start_time']
    batches: list[Record] = []
    product_count = 0
    if settings['arrival_rate'] == 0:
        return batches
    while True:
        interval = -math.log1p(-generator.random()) / settings['arrival_rate']
        next_time = time + interval
        if next_time <= time:
            raise ValueError('Arrival interval does not advance the floating-point clock')
        time = next_time
        if time > settings['end_time']:
            return batches
        distribution = settings['batch_size']
        size = distribution if type(distribution) is int else distribution['values'][weighted_index(generator, distribution['weights'])]
        product_ids = [f'P{index:06d}' for index in range(product_count + 1, product_count + size + 1)]
        batches.append({'id': f'B{len(batches) + 1:06d}', 'release_time': time,
                        'batch_size': size, 'product_ids': product_ids})
        product_count += size


def generate_products(configuration: Record, batches: list[Record], generator: random.Random) -> list[Record]:
    """Sample independent qualities from the filtered, renormalized eligible classes for every item."""
    overrides = configuration['collection_point'].get('quality_weights', {})
    qualities = sorted((q for q in configuration['quality_classes']
        if configuration['recycling_facilities'] or q['disposition'] == 'remanufacture'), key=lambda quality: quality['id'])
    weights = [overrides.get(quality['id'], quality['weight']) for quality in qualities]
    total = sum(weights)
    probabilities = [weight / total for weight in weights] if total else []
    products = []
    for batch in batches:
        for product_id in batch['product_ids']:
            quality = qualities[weighted_index(generator, probabilities)]
            products.append({'id': product_id, 'batch_id': batch['id'], 'origin': configuration['collection_point']['id'],
                'release_time': batch['release_time'], 'quality_class': quality['id'],
                'disposition': quality['disposition'], 'operations': list(quality['operations'])})
    return products


def prepare_scenario(configuration_path: Path, seed: int) -> tuple[Record, bytes]:
    """Resolve one configuration into canonical LF scenario bytes with products and seed provenance."""
    if type(seed) is not int:
        raise ValueError('Generation requires an explicit integer seed')
    configuration = json.loads(configuration_path.read_text(encoding='utf-8-sig'))
    if configuration['format_version'] == 3:
        from .network_generation import prepare_network_scenario
        return prepare_network_scenario(configuration, seed)
    validate_generation_config(configuration)
    stream_seeds = {name: derive_stream_seed(seed, name) for name in ('arrivals_batches', 'qualities')}
    batches = generate_return_batches(configuration, random.Random(stream_seeds['arrivals_batches']))
    products = generate_products(configuration, batches, random.Random(stream_seeds['qualities']))
    scenario = copy.deepcopy(configuration)
    scenario.update(products=products, arrival_batches=batches, seed_provenance={
        'source': 'seeded_configuration', 'master_seed': seed, 'generation_version': 1,
        'algorithm': 'Random.random/exponential-inverse-cdf/weighted-cdf-v1', 'stream_seeds': stream_seeds})
    validate_inventory_scenario(scenario)
    scenario_bytes = (json.dumps(scenario, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    return json.loads(scenario_bytes), scenario_bytes
