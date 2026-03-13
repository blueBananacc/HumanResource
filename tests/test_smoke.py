from human_resource.main import run


def test_smoke(capsys):
    run()
    captured = capsys.readouterr()
    assert "HumanResource project is ready." in captured.out
