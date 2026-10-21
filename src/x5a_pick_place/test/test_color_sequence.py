import pytest
from x5a_pick_place.color_sequence import color_step_pairs


def step(source, destination):
    return {'action':'pick_place','target_color':source,'destination_color':destination}


def test_repeated_source_and_swapping_support_are_valid():
    assert color_step_pairs([step('white','orange'),step('white','red')]) == [('white','orange'),('white','red')]
    assert color_step_pairs([step('white','orange'),step('orange','white')]) == [('white','orange'),('orange','white')]


def test_hand_transitions_and_bound_queue():
    assert color_step_pairs([{'action':'pick','target_color':'white'},
                             {'action':'place','destination_color':'red'}]) == [('white','hold'),('white','red')]
    assert color_step_pairs([step('white','orange'),step('white','red')], 'white')
    assert len(color_step_pairs([step('white','box')]*8)) == 8


@pytest.mark.parametrize('tasks,held', [([],None),([step('white','box')]*9,None),
    ([step('white','white')],None),([step('white','orange')],'red'),
    ([step('white','orange'),step('red','orange')],None),
    ([{'action':'pick','target_color':'red'},step('white','orange')],None),
    ([{'action':'place','destination_color':'red'}],None),
    ([step('blue','orange')],None),([{'action':'release'}],None)])
def test_invalid_whole_program(tasks,held):
    with pytest.raises(ValueError):
        color_step_pairs(tasks,held)


def test_build_three_layers_then_take_the_bottom_and_rebuild():
    tasks=[step('white','orange'),step('red','white'),step('orange','white'),step('red','orange')]
    assert color_step_pairs(tasks) == [('white','orange'),('red','white'),('orange','white'),('red','orange')]
