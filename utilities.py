def remove_duplicates_from_dict_values(data: dict) -> dict:
    """
    Removes duplicate values from the lists of each key in a dictionary (in-place).
    Args:
        data (dict): Dictionary with lists as values.
    Returns:
        dict: The same dictionary with duplicate values removed from each list.
    """
    for key, values in data.items():
        seen = set()
        unique_values = []
        for v in values:
            if v not in seen:
                seen.add(v)
                unique_values.append(v)
        # Modify the list in-place
        data[key][:] = unique_values
    return data


def remove_duplicates_from_list_values(words: list) -> list:
    """
    Removes duplicate values from a list (in-place) while preserving order.
    Args:
        words (list): The input list of strings.
    Returns:
        list: The same list object with duplicates removed.
    """
    seen = set()
    unique_words = []
    for word in words:
        if word not in seen:
            seen.add(word)
            unique_words.append(word)
    # Modify the original list in place
    words[:] = unique_words
    return words
