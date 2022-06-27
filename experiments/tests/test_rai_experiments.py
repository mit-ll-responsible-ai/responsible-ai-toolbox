def test_version():
    import rai_experiments

    assert isinstance(rai_experiments.__version__, str)
    assert rai_experiments.__version__.count(".") == 2


def test_pre_trained_models():
    from rai_experiments.models.pretrained import load_model

    out = load_model("mitll_cifar_l2_1_0.pt")
    out.eval()


def test_imports():
    from rai_experiments import utils
    from rai_experiments.models import pretrained, resnet, small_resnet

    assert utils
    assert pretrained
    assert resnet
    assert small_resnet
