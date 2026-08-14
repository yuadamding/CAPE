# ADR 0001: sibling repository

Decision: implement count-SDE V4 in an independently versioned sibling
distribution and leave the frozen CREDO checkout byte-identical.

Numerical internals are import-isolated. Supported lifecycle calls consume the
exact frozen compatibility receipt. This keeps retained V2/V3 implementation
hashes and loaders untouched.
