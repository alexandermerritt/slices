#include "../lib.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Error Handling
//===----------------------------------------------------------------------===//

const char* cudaGetErrorString(cudaError_t error)
{
#if defined(TIMING) && defined(TIMING_NATIVE)
	return bypass.cudaGetErrorString(error);
#else
	typedef const char*(*func_t)(cudaError_t);
	func_t func = (func_t)dlsym(RTLD_NEXT, __func__);
	if (func) return func(error);
	switch (error) {
		case cudaSuccess: return "cudaSuccess";
		case cudaErrorMissingConfiguration: return "cudaErrorMissingConfiguration";
		case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
		case cudaErrorInitializationError: return "cudaErrorInitializationError";
		case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
		case cudaErrorPriorLaunchFailure: return "cudaErrorPriorLaunchFailure";
		case cudaErrorLaunchTimeout: return "cudaErrorLaunchTimeout";
		case cudaErrorLaunchOutOfResources: return "cudaErrorLaunchOutOfResources";
		case cudaErrorInvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction";
		case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration";
		case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
		case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
		case cudaErrorInvalidPitchValue: return "cudaErrorInvalidPitchValue";
		case cudaErrorInvalidSymbol: return "cudaErrorInvalidSymbol";
		case cudaErrorMapBufferObjectFailed: return "cudaErrorMapBufferObjectFailed";
		case cudaErrorUnmapBufferObjectFailed: return "cudaErrorUnmapBufferObjectFailed";
		case cudaErrorInvalidHostPointer: return "cudaErrorInvalidHostPointer";
		case cudaErrorInvalidDevicePointer: return "cudaErrorInvalidDevicePointer";
		case cudaErrorInvalidTexture: return "cudaErrorInvalidTexture";
		case cudaErrorInvalidTextureBinding: return "cudaErrorInvalidTextureBinding";
		case cudaErrorInvalidChannelDescriptor: return "cudaErrorInvalidChannelDescriptor";
		case cudaErrorInvalidMemcpyDirection: return "cudaErrorInvalidMemcpyDirection";
		case cudaErrorAddressOfConstant: return "cudaErrorAddressOfConstant";
		case cudaErrorTextureFetchFailed: return "cudaErrorTextureFetchFailed";
		case cudaErrorTextureNotBound: return "cudaErrorTextureNotBound";
		case cudaErrorSynchronizationError: return "cudaErrorSynchronizationError";
		case cudaErrorInvalidFilterSetting: return "cudaErrorInvalidFilterSetting";
		case cudaErrorInvalidNormSetting: return "cudaErrorInvalidNormSetting";
		case cudaErrorMixedDeviceExecution: return "cudaErrorMixedDeviceExecution";
		case cudaErrorCudartUnloading: return "cudaErrorCudartUnloading";
		case cudaErrorNotYetImplemented: return "cudaErrorNotYetImplemented";
		case cudaErrorMemoryValueTooLarge: return "cudaErrorMemoryValueTooLarge";
		case cudaErrorInvalidResourceHandle: return "cudaErrorInvalidResourceHandle";
		case cudaErrorNotReady: return "cudaErrorNotReady";
		case cudaErrorInsufficientDriver: return "cudaErrorInsufficientDriver";
		case cudaErrorSetOnActiveProcess: return "cudaErrorSetOnActiveProcess";
		case cudaErrorInvalidSurface: return "cudaErrorInvalidSurface";
		case cudaErrorNoDevice: return "cudaErrorNoDevice";
		case cudaErrorECCUncorrectable: return "cudaErrorECCUncorrectable";
		case cudaErrorSharedObjectSymbolNotFound: return "cudaErrorSharedObjectSymbolNotFound";
		case cudaErrorSharedObjectInitFailed: return "cudaErrorSharedObjectInitFailed";
		case cudaErrorUnsupportedLimit: return "cudaErrorUnsupportedLimit";
		case cudaErrorDuplicateVariableName: return "cudaErrorDuplicateVariableName";
		case cudaErrorDuplicateTextureName: return "cudaErrorDuplicateTextureName";
		case cudaErrorDuplicateSurfaceName: return "cudaErrorDuplicateSurfaceName";
		case cudaErrorDevicesUnavailable: return "cudaErrorDevicesUnavailable";
		case cudaErrorInvalidKernelImage: return "cudaErrorInvalidKernelImage";
		case cudaErrorNoKernelImageForDevice: return "cudaErrorNoKernelImageForDevice";
		case cudaErrorIncompatibleDriverContext: return "cudaErrorIncompatibleDriverContext";
		case cudaErrorStartupFailure: return "cudaErrorStartupFailure";
		case cudaErrorApiFailureBase: return "cudaErrorApiFailureBase";
		case cudaErrorUnknown:
		default:
			return "cudaErrorUnknown";
	}
#endif
}

cudaError_t cudaGetLastError(void)
{
#if defined(TIMING) && defined(TIMING_NATIVE)
	return bypass.cudaGetLastError();
#else
	return cuda_err; // ??
#endif
}

