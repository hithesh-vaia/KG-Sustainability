from eval.normalize import contains_all, contains_any


def test_alternatives_any_of():
    # a nested list means "any of these phrasings is acceptable"
    needles = [["none", "zero", "nil"]]
    for ans in ["There were none.", "Zero fatalities were reported.", "Nil."]:
        ok, missing = contains_all(ans, needles)
        assert ok, (ans, missing)
    ok, missing = contains_all("There were 14 incidents.", needles)
    assert not ok and missing == ["none | zero | nil"]


def test_plain_strings_still_work():
    assert contains_all("Scope 2 was 57,763.14 tCO2e", ["57763.14", "tCO2e"])[0]
    assert contains_any("we disclose a financed emissions total", ["financed emissions total"])


def test_alternatives_in_must_not_include():
    assert contains_any("the total is 1,234 tCO2e", [["tco2e", "tonnes"]]) == ["tco2e | tonnes"]


def test_incidental_digits_do_not_match_as_numbers():
    # "tCO2e ..." contains a 2; it must not match an answer merely containing 2
    ans = "There were 2 incidents reported in the quarter."
    assert contains_any(ans, ["tCO2e for the entire loan portfolio"]) == []
    assert contains_any("we saw 45001 widgets", ["ISO 45001:2018"]) == []
    # accepted limitation: a needle that *starts* with a digit ("24x7 SOC") is
    # still compared numerically; only label-leading needles are excluded.


def test_real_numeric_needles_still_match_with_tolerance():
    assert contains_all("Scope 2 emissions were 57,763.14 tCO2e", ["57763.14"])[0]
    assert contains_all("total was 3,10,573.94 GJ", ["310573.94"])[0]   # Indian grouping
    assert contains_all("reported 4.30 MT of plastic", ["4.30 MT"])[0]
