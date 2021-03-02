import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.distribution.UniformDistribution
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Sgd

fun main() {
    val input = Nd4j.zeros(4,2)
    val labels = Nd4j.zeros(4,2)

    input.putScalar(intArrayOf(0,0), 0)
    input.putScalar(intArrayOf(0,1), 0)
    input.putScalar(intArrayOf(1,0), 1)
    input.putScalar(intArrayOf(1,1), 0)
    input.putScalar(intArrayOf(2,0), 0)
    input.putScalar(intArrayOf(2,1), 1)
    input.putScalar(intArrayOf(3,0), 0)
    input.putScalar(intArrayOf(3,1), 1)

    labels.putScalar(intArrayOf(0,0), 0)
    labels.putScalar(intArrayOf(0,1), 1)
    labels.putScalar(intArrayOf(1,0), 1)
    labels.putScalar(intArrayOf(1,1), 0)
    labels.putScalar(intArrayOf(2,0), 0)
    labels.putScalar(intArrayOf(2,1), 1)
    labels.putScalar(intArrayOf(3,0), 0)
    labels.putScalar(intArrayOf(3,1), 1)

    val set = DataSet(input, labels)

    val builder = NeuralNetConfiguration.Builder().apply {
        this.seed = 123
        this.optimizationAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
        this.biasInit = 0.0
        this.isMiniBatch = false
        this.updater(Sgd(0.1))
    }

    val denseLayer = DenseLayer.Builder().apply {
        this.nIn = 2
        this.nOut = 4
        this.activation(Activation.RELU)
        this.weightInit(UniformDistribution(0.0, 1.0))
    }

    val denseLayer2 = DenseLayer.Builder().apply {
        this.nIn = 4
        this.nOut = 4
        this.activation(Activation.RELU)
        this.weightInit(UniformDistribution(0.0, 1.0))
    }

    val output = OutputLayer.Builder().apply {
        this.nIn = 4
        this.nOut = 2
        this.activation(Activation.SOFTMAX)
        this.weightInit(UniformDistribution(0.0, 1.0))
    }

    val listBuilder = builder.list()
    listBuilder.layer(0, denseLayer.build())
    listBuilder.layer(1, denseLayer2.build())
    listBuilder.layer(2, output.build())

    val netConfig = listBuilder.build()
    val net = MultiLayerNetwork(netConfig)
    net.setListeners(ScoreIterationListener(100))
    net.init()
    for (i in 0..100000) {
        net.fit(set)
    }

    val result = net.output(set.features)

    val eval = Evaluation(2)
    eval.eval(set.labels, result)
    println(eval.stats())

    println("Predicting...")
    println(net.output(input))
}