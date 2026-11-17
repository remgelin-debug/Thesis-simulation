"""Seeded workstation windows, independent of return and quality sampling."""
import math
import random
from .models import Record


def validate_schedule_generation(settings: Record, horizon: float) -> None:
    """Require bounded observation times and finite ranges that always advance the calendar."""
    if not all(key in settings for key in ('start_time', 'end_time', 'initial_delay', 'open_duration', 'closed_duration')):
        raise ValueError('Schedule settings require start_time, end_time, initial_delay, open_duration and closed_duration')
    beginning, end = settings['start_time'], settings['end_time']
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in (beginning, end)) or not 0 <= beginning <= end <= horizon:
        raise ValueError('Schedule generation requires 0 <= start_time <= end_time <= horizon')
    for name in ('initial_delay', 'open_duration', 'closed_duration'):
        interval = settings[name]
        if not isinstance(interval, list) or len(interval) != 2 or any(type(value) not in (int, float) or not math.isfinite(value) for value in interval):
            raise ValueError(f'Schedule {name} must contain two finite bounds')
        lower, upper = interval
        if not 0 <= lower <= upper or (name != 'initial_delay' and lower <= 0):
            raise ValueError(f'Schedule {name} bounds must be ordered and durations positive')


def generate_windows(settings: Record, generator: random.Random) -> list[list[float]]:
    """Sample initial delay and alternating open/closed durations, clipping the last window at end_time."""
    opening = settings['start_time'] + generator.uniform(*settings['initial_delay'])
    windows = []
    while opening < settings['end_time']:
        closing = min(settings['end_time'], opening + generator.uniform(*settings['open_duration']))
        if closing <= opening:
            raise ValueError('Sampled opening duration cannot advance the floating-point clock')
        windows.append([opening, closing])
        if closing == settings['end_time']:
            break
        opening = closing + generator.uniform(*settings['closed_duration'])
        if opening <= closing:
            raise ValueError('Sampled closure duration cannot advance the floating-point clock')
    return windows


def station_schedule_settings(configuration: Record, station: Record) -> Record | None:
    """Merge global schedule parameters with station overrides, or preserve explicit fixed windows."""
    if 'windows' in station:
        if 'window_generation' in station:
            raise ValueError('Choose fixed windows or window_generation for a workstation, not both')
        return None
    settings = {**configuration.get('schedule_generation', {}), **station.get('window_generation', {})}
    validate_schedule_generation(settings, configuration['horizon'])
    return settings
