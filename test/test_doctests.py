"""Execute the doctest examples carried by the public `scs` API.

Docstring *coverage* only says a docstring exists; this says the examples
inside it still evaluate to what they claim. Runs against the installed
package, which is the one users import.
"""

import doctest

import scs


def test_docstring_examples_run(capsys):
    """Every `>>>` example in `scs` still produces its documented output."""
    results = doctest.testmod(
        scs,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS,
        verbose=False,
    )
    report = capsys.readouterr().out

    # A module that has lost its examples would otherwise pass silently.
    assert results.attempted > 0, "no doctest examples found in scs"
    assert results.failed == 0, report
