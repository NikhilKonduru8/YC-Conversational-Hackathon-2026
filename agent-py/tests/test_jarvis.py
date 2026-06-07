"""Tests for the sentence splitter that drives low-latency streaming TTS."""

from nodes.reasoning_node import _pop_sentence


def test_pops_first_complete_sentence():
    s, rest = _pop_sentence("Five volts. And ")
    assert s == "Five volts."
    assert rest == " And "


def test_no_complete_sentence_yet():
    s, rest = _pop_sentence("the maximum is")
    assert s is None
    assert rest == "the maximum is"


def test_handles_question_and_exclamation():
    s, rest = _pop_sentence("Is it lit? Yes")
    assert s == "Is it lit?"
    assert rest == " Yes"


def test_period_at_very_end():
    s, rest = _pop_sentence("Pin three is the output.")
    assert s == "Pin three is the output."
    assert rest == ""


def test_decimal_point_is_not_a_sentence_break():
    # "3.3" should not split because the '.' is followed by a digit, not a space.
    s, _ = _pop_sentence("It is 3.3 volts")
    assert s is None
