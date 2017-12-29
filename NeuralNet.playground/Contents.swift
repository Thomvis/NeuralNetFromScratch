import Foundation

// References:
// https://stackoverflow.com/a/43315365/611472
// https://github.com/Swift-AI/NeuralNet/blob/master/Sources/NeuralNet.swift
// https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/
// http://machinethink.net/blog/the-hello-world-of-neural-networks/
// http://blog.karmadust.com/starting-with-neural-networks-in-swift-3-0/
// https://matrices.io/deep-neural-network-from-scratch/

protocol ActivationFunction {
    static func y(_ x: Double) -> Double
    static func diff(_ y: Double) -> Double
}

struct Sigmoid: ActivationFunction {
    static func y(_ x: Double) -> Double {
        return 1.0 / (1.0 + exp(-x))
    }
    
    static func diff(_ y: Double) -> Double {
        return y * (1.0 - y)
    }

}

protocol Layer {
    var neuronCount: Int { get }
    var activationFunction: ActivationFunction.Type { get }
    
    var weights: [Double] { get set }
    
    func output(_ input: [Double]) -> [Double]
}

struct InputLayer: Layer {
    
    let neuronCount: Int
    var activationFunction: ActivationFunction.Type {
        fatalError()
    }
    let providesBias: Bool
    
    var weights: [Double]
    
    init(neuronCount: Int, provideBias: Bool) {
        self.neuronCount = neuronCount
        self.weights = Array(repeating: 0.0, count: neuronCount)
        self.providesBias = true
    }
    
    func output(_ input: [Double]) -> [Double] {
        return input + (self.providesBias ? [1.0] : [])
    }
}

struct FullyConnectedLayer: Layer {
    let neuronCount: Int
    let activationFunction: ActivationFunction.Type
    let providesBias: Bool
    
    // weights applied to the output values of the previous layer
    // the connection between neuron n of the previous and m of this layer
    // has weight at index
    var weights: [Double]
    
    init(neuronCount: Int, previous: Layer, weights: [Double]? = nil, provideBias: Bool) {
        self.neuronCount = neuronCount
        self.activationFunction = Sigmoid.self
        self.providesBias = provideBias
        
        // Initializing all weights to 0.0
        self.weights = weights ?? Array(repeating: 0.0, count: previous.neuronCount*neuronCount)
    }
    
    func output(_ input: [Double]) -> [Double] {
        var output: [Double] = []
        print("\t\tInput: \(input)")
        for n in 0..<neuronCount {
            print("\t\tNeuron \(n)")
            let str = Array(stride(from: n, to: self.weights.count, by: neuronCount))
            let weights = str.map { self.weights[$0] }
            print("\t\t\tWeights: \(weights)")
            let sum = zip(input, weights).reduce(0.0) { sum, iw in
                return sum + iw.0*iw.1
            }
            let act = activationFunction.y(sum)
            print("\t\t\tSum: \(sum), activation: \(act)")
            output.append(act)
        }
        return output + (self.providesBias ? [1.0] : [])
    }
}

struct NeuralNet {
    var layers: [Layer]
    
    init(inputNeuronCount: Int, hiddenNeuronCounts: [Int], outputNeuronCount: Int) {
        self.layers = []
        
        // input layer
        var prev: Layer = InputLayer(neuronCount: inputNeuronCount, provideBias: true)
        self.layers.append(prev)
        
        // hidden layers
        for c in hiddenNeuronCounts {
            prev = FullyConnectedLayer(neuronCount: c, previous: prev, weights: nil, provideBias: true)
            self.layers.append(prev)
        }
        
        // output layer
        self.layers.append(FullyConnectedLayer(
            neuronCount: outputNeuronCount,
            previous: prev,
            weights: nil,
            provideBias: false
        ))
    }
}

extension NeuralNet {
    func infer(_ input: [Double]) -> [Double] {
        return inferWithIntermediates(input).last!
    }
    
    func inferWithIntermediates(_ input: [Double]) -> [[Double]] {
        var output = [input]
        print("Inferring \(input)")
        for (i, l) in layers.enumerated() {
            print("\tLayer \(i)")
            output.append(l.output(output.last!))
        }
        return output
    }
}

// Create the net
var net = NeuralNet(
    inputNeuronCount: 2,
    hiddenNeuronCounts: [2],
    outputNeuronCount: 1
)

// Set 'magic' weights & biases
net.layers[1].weights = [54, 14, 17, 14, -8, -20]
net.layers[2].weights = [92, -92, -48]


net.infer([0.0, 0.0])
net.infer([1.0, 0.0])
net.infer([0.0, 1.0])
net.infer([1.0, 1.0])

// Training
//extension NeuralNet {
//    mutating func train(input: [Double], expected: [Double]) {
//        let inferred = inferWithIntermediates(input)
//
//        // Calculate gradient of error at output layer
//        var error = zip(inferred.last!, expected).map { $0.0 - $0.1 }
//
//        // Iterate over layers, starting with the output layer and ignoring the input layer
//        for i in (1..<layers.count).reversed() {
//            let layer = layers[i]
//            let layerOutput = inferred[i]
//
//            let slope = layerOutput.map(layer.activationFunction.diff)
//            let delta = zip(error, slope).map(*)
//        }
//    }
//}

//net.train(input: [0.0, 1.0], expected: [1.0])

//net.infer([0.0, 0.0])
//net.infer([1.0, 0.0])
//net.infer([0.0, 1.0])
//net.infer([1.0, 1.0])

