from scripts.split_events import event_key


def test_event_key_groups_zone_views():
    a = "/example/2000-01-01-00-00-00-0_x_y_data_1-1.0-5.0.npy"
    b = "/example/2000-01-01-00-00-00-0_x_y_data_3-1.0-5.0.npy"
    assert event_key(a) == event_key(b) == "2000-01-01-00-00-00-0"
