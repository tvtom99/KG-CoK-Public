"""
    Author: Thomas Marshall Childs
    Creation Date: 17/09/2025
    Purpose:
        This file is intended to hold any typing definitions that are used within the framework.
"""

from typing import TypeAlias

Knowledge: TypeAlias = dict[str, dict[str, str]];
"""
A dictionary of the form:
    {
        domain1: {
            source1: information,
            sourceN: information
        },
        domainN: {
            source1: information,
            sourceN: information
        }
    }
"""
