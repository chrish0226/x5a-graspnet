"""A two-step place command must locate its destination, not the held object."""
import asyncio

import pytest
from x5a_web_agent import app


@pytest.mark.parametrize('text,destination,relation', [
    ('放到盒子上。', 'box', 'on'),
    ('放进黑色盒子里', 'black box', 'in'),
    ('放到白色方块上', 'white cube', 'on'),
])
def test_place_destination_precedes_held_object(monkeypatch, text, destination, relation):
    monkeypatch.setenv('X5A_GRASP_ROUTE', 'remote')
    monkeypatch.setenv('X5A_CLOUD_PARSE', '1')
    monkeypatch.setattr(app, 'cloud_parse_text', lambda _: dict(
        object='cube', color='red', sam_prompt='red cube', action='place',
        place_prompt=destination, place_rel=relation))
    parsed = asyncio.run(app._cloud_parse_command(text))
    assert parsed['action'] == 'remote_place'
    assert parsed['prompt'] == destination
    assert parsed['relation'] == relation


@pytest.mark.parametrize('missing', [None, 'null', ''])
def test_legacy_place_response_uses_sam_prompt_when_destination_absent(monkeypatch, missing):
    monkeypatch.setenv('X5A_GRASP_ROUTE', 'remote')
    monkeypatch.setenv('X5A_CLOUD_PARSE', '1')
    monkeypatch.setattr(app, 'cloud_parse_text', lambda _: dict(
        object='box', sam_prompt='box', action='place', place_prompt=missing))
    parsed = asyncio.run(app._cloud_parse_command('放到盒子上'))
    assert parsed['prompt'] == 'box'


def test_unavailable_cloud_keeps_spoken_place_target(monkeypatch):
    monkeypatch.setenv('X5A_GRASP_ROUTE', 'remote')
    monkeypatch.setenv('X5A_CLOUD_PARSE', '1')
    monkeypatch.setattr(app, 'cloud_parse_text', lambda _: None)
    parsed = asyncio.run(app._cloud_parse_command('放到盒子上'))
    assert parsed['prompt'] == '盒子'
