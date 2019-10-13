///
///  Anime4KMetal.Playground
///
///  Copyright © 2019 Marcus Zhou. All rights reserved.
///
///  Anime4KMetalPlayground is free software: you can redistribute it and/or
///  modify it under the terms of the GNU General Public License as published
///  by the Free Software Foundation, either version 3 of the License, or
///  (at your option) any later version.
///
///  Anime4KMetalPlayground is distributed in the hope that it will be useful,
///  but WITHOUT ANY WARRANTY; without even the implied warranty of
///  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
///  GNU General Public License for more details.
///
///  You should have received a copy of the GNU General Public License
///  along with NineAnimator.  If not, see <http://www.gnu.org/licenses/>.
///

import Cocoa
import Metal
import MetalKit

enum XError: Error {
    case internalError
}

// MARK: - Helper Methods
extension CGImage {
    func toRGBBitmapArray() throws -> [UInt8] {
        let cgColorSpaceDevice = CGColorSpaceCreateDeviceRGB()
        var buffer = [UInt8](
            repeating: 0x00,
            count: bytesPerRow * height
        )
        
        guard let context = CGContext(
            data: &buffer,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: cgColorSpaceDevice,
            bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { throw XError.internalError }
        
        // Draw the image in the context
        context.draw(
            self,
            in: .init(
                origin: .zero,
                size: .init(width: width, height: height)
            )
        )
        
        return buffer
    }
    
    /// Obtain the number of bytes per row
    var bytesPerRow: Int {
        return bytesPerPixel * width
    }
    
    /// Obtain the number of bytes per pixel
    var bytesPerPixel: Int {
        return bitsPerPixel / bitsPerComponent
    }
}

func makeTexture(_ device: MTLDevice, width: Int, height: Int) -> MTLTexture {
    let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm,
        width: width,
        height: height,
        mipmapped: false
    )
    return device.makeTexture(descriptor: textureDescriptor)!
}

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Unable to get the default Metal-capable Device")
}

let sourceUrl = Bundle.main.url(forResource: "shaders/Anime4KMetal", withExtension: "metal")!
let source = try! String(contentsOf: sourceUrl)
let library = try! device.makeLibrary(source: source, options: nil)
let commandQueue = device.makeCommandQueue()!
let commandBuffer = commandQueue.makeCommandBuffer()!

// Read Images
let inputImage = NSImage(named: "1080p_image")!
let cgInputImage = inputImage.cgImage(forProposedRect: nil, context: nil, hints: nil)!

// MARK: - Configurations
var scaleFactor: Float = 2.0
var srcHeight = cgInputImage.height
var srcWidth = cgInputImage.width
var dstHeight = Int(Float(srcHeight) * scaleFactor)
var dstWidth = Int(Float(srcWidth) * scaleFactor)
var srcBytesPerPixel = cgInputImage.bitsPerPixel / cgInputImage.bitsPerComponent
var srcBytesPerRow = srcBytesPerPixel * srcWidth
var dstBytesPerRow = Int(Float(srcBytesPerRow) * scaleFactor)
var bold: Float = 10
var blur: Float = 0.0001
let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
let cgColorSpaceDevice = CGColorSpaceCreateDeviceRGB()

// Convert Image to RGBA bitmap
var inputImageBitmap = try! cgInputImage.toRGBBitmapArray()

// Configure Constants

let kernalConfiguration = MTLFunctionConstantValues()
kernalConfiguration.setConstantValue(&scaleFactor, type: .float, index: 0)
kernalConfiguration.setConstantValue(&srcWidth, type: .uint, index: 1)
kernalConfiguration.setConstantValue(&srcHeight, type: .uint, index: 2)
kernalConfiguration.setConstantValue(&dstWidth, type: .uint, index: 3)
kernalConfiguration.setConstantValue(&dstHeight, type: .uint, index: 4)
kernalConfiguration.setConstantValue(&dstBytesPerRow, type: .uint, index: 5)
kernalConfiguration.setConstantValue(&bold, type: .float, index: 6)
kernalConfiguration.setConstantValue(&blur, type: .float, index: 7)

// Thread groups: 1 per ouput pixel
let threadGroupCount = MTLSize(width: 1, height: 1, depth: 1)
let threadGroup = MTLSize(
    width: dstWidth / threadGroupCount.width,
    height: dstHeight / threadGroupCount.height,
    depth: 1
)

func dumpTexture(_ texture: MTLTexture) -> CGImage {
    // Export Converted Images
    var dstImageBitmap = [UInt8](
        repeating: 0x00,
        count: dstBytesPerRow * dstHeight
    )
    texture.getBytes(
        &dstImageBitmap,
        bytesPerRow: dstBytesPerRow,
        from: MTLRegionMake2D(0, 0, dstWidth, dstHeight),
        mipmapLevel: 0
    )
    
    print("Bitmap Length: \(dstImageBitmap.count), bytesPerRow: \(srcBytesPerRow), size: (\(dstWidth), \(dstHeight))", texture)

    let dstProvider = CGDataProvider(data: Data(bytes: &dstImageBitmap, count: dstImageBitmap.count) as CFData)!
    return CGImage(
        width: dstWidth,
        height: dstHeight,
        bitsPerComponent: cgInputImage.bitsPerComponent,
        bitsPerPixel: cgInputImage.bitsPerPixel,
        bytesPerRow: dstBytesPerRow,
        space: cgColorSpaceDevice,
        bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
        provider: dstProvider,
        decode: nil,
        shouldInterpolate: false,
        intent: .defaultIntent
    )!
}

// MARK: - Scaling Image

// Convert Bitmap to Texture
let srcTexture = makeTexture(device, width: srcWidth, height: srcHeight)
srcTexture.replace(
    region: MTLRegionMake2D(0, 0, srcWidth, srcHeight),
    mipmapLevel: 0,
    withBytes: &inputImageBitmap,
    bytesPerRow: srcBytesPerRow
)

//// Create output texture
let scaledTexture = makeTexture(device, width: dstWidth, height: dstHeight)

// Build & run kernel
let scalingKernalFunction = try! library.makeFunction(name: "ScaleMain", constantValues: kernalConfiguration)
let scalingKernalPipelineState = try! device.makeComputePipelineState(function: scalingKernalFunction)
let scalingCommandEncoder = commandBuffer.makeComputeCommandEncoder()!

scalingCommandEncoder.setComputePipelineState(scalingKernalPipelineState)
scalingCommandEncoder.setTexture(srcTexture, index: 0)
scalingCommandEncoder.setTexture(scaledTexture, index: 1)

scalingCommandEncoder.dispatchThreadgroups(threadGroup, threadsPerThreadgroup: threadGroupCount)
scalingCommandEncoder.endEncoding()

//commandBuffer.commit()
//commandBuffer.waitUntilCompleted()

// MARK: - Luminance

let lumTexture = makeTexture(device, width: dstWidth, height: dstHeight)

// Build & run kernel
let lumKernalFunction = try! library.makeFunction(name: "LumMain", constantValues: kernalConfiguration)
let lumKernalPipelineState = try! device.makeComputePipelineState(function: lumKernalFunction)
let lumCommandEncoder = commandBuffer.makeComputeCommandEncoder()!

lumCommandEncoder.setComputePipelineState(lumKernalPipelineState)
lumCommandEncoder.setTexture(scaledTexture, index: 0)
lumCommandEncoder.setTexture(lumTexture, index: 1)

lumCommandEncoder.dispatchThreadgroups(threadGroup, threadsPerThreadgroup: threadGroupCount)
lumCommandEncoder.endEncoding()

// MARK: - Push

let pushTexture = makeTexture(device, width: dstWidth, height: dstHeight)

// Build & run kernel
let pushKernalFunction = try! library.makeFunction(name: "PushMain", constantValues: kernalConfiguration)
let pushKernalPipelineState = try! device.makeComputePipelineState(function: pushKernalFunction)
let pushCommandEncoder = commandBuffer.makeComputeCommandEncoder()!

pushCommandEncoder.setComputePipelineState(pushKernalPipelineState)
pushCommandEncoder.setTexture(scaledTexture, index: 0)
pushCommandEncoder.setTexture(lumTexture, index: 1)
pushCommandEncoder.setTexture(pushTexture, index: 2)

pushCommandEncoder.dispatchThreadgroups(threadGroup, threadsPerThreadgroup: threadGroupCount)
pushCommandEncoder.endEncoding()

// MARK: - Lum of Pushed Image

let lumPushTexture = makeTexture(device, width: dstWidth, height: dstHeight)

// Build & run kernel
let lumPushKernalFunction = try! library.makeFunction(name: "LumMain", constantValues: kernalConfiguration)
let lumPushKernalPipelineState = try! device.makeComputePipelineState(function: lumPushKernalFunction)
let lumPushCommandEncoder = commandBuffer.makeComputeCommandEncoder()!

lumPushCommandEncoder.setComputePipelineState(lumPushKernalPipelineState)
lumPushCommandEncoder.setTexture(pushTexture, index: 0)
lumPushCommandEncoder.setTexture(lumPushTexture, index: 1)

lumPushCommandEncoder.dispatchThreadgroups(threadGroup, threadsPerThreadgroup: threadGroupCount)
lumPushCommandEncoder.endEncoding()

// MARK: - Gradient

let gradTexture = makeTexture(device, width: dstWidth, height: dstHeight)

// Build & run kernel
let gradKernalFunction = try! library.makeFunction(name: "GradMain", constantValues: kernalConfiguration)
let gradKernalPipelineState = try! device.makeComputePipelineState(function: gradKernalFunction)
let gradCommandEncoder = commandBuffer.makeComputeCommandEncoder()!

gradCommandEncoder.setComputePipelineState(gradKernalPipelineState)
gradCommandEncoder.setTexture(pushTexture, index: 0)
gradCommandEncoder.setTexture(lumPushTexture, index: 1)
gradCommandEncoder.setTexture(gradTexture, index: 2)

gradCommandEncoder.dispatchThreadgroups(threadGroup, threadsPerThreadgroup: threadGroupCount)
gradCommandEncoder.endEncoding()

// MARK: - Final

let finalTexture = makeTexture(device, width: dstWidth, height: dstHeight)

// Build & run kernel
let finalKernalFunction = try! library.makeFunction(name: "FinalMain", constantValues: kernalConfiguration)
let finalKernalPipelineState = try! device.makeComputePipelineState(function: finalKernalFunction)
let finalCommandEncoder = commandBuffer.makeComputeCommandEncoder()!

finalCommandEncoder.setComputePipelineState(finalKernalPipelineState)
finalCommandEncoder.setTexture(pushTexture, index: 0)
finalCommandEncoder.setTexture(gradTexture, index: 1)
finalCommandEncoder.setTexture(scaledTexture, index: 2)
finalCommandEncoder.setTexture(finalTexture, index: 3)

finalCommandEncoder.dispatchThreadgroups(threadGroup, threadsPerThreadgroup: threadGroupCount)
finalCommandEncoder.endEncoding()

// MARK: - Execute & Results

commandBuffer.commit()
commandBuffer.waitUntilCompleted()

// Scaled Image
dumpTexture(scaledTexture)
inputImage
// Final Product
dumpTexture(finalTexture)

// Luminance
dumpTexture(lumTexture)

// Push
dumpTexture(pushTexture)

// Lum of Pushed
dumpTexture(lumPushTexture)

// Gradient
dumpTexture(gradTexture)
