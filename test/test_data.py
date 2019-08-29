from src.data import read_raw_data, preprocess_data, get_features, get_target


def test_raw_shape():
    dframe = read_raw_data()
    assert dframe.shape == (34109, 8)


def test_get_features_shape():
    dframe = read_raw_data()
    processed = preprocess_data(dframe)
    features = get_features(processed)
    label = get_target(processed)

    assert features.shape == (34109, 7)
    assert label.shape == (34109)