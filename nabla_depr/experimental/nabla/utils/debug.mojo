from builtin._location import __call_location


@always_inline
fn err_loc() -> String:
    return "\n" + String(__call_location())
