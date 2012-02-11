/**
 * @file fatcubininfo.h
 * @brief Defines data relate to fat cubin binary; data copied from
 * remote_gpu/nvidia_backend/defs.h
 *
 * @date Mar 21, 2011
 * @author Magda Slawinska, magg __at_ gatech __dot_ edu
 *
 * @date 2011-11-04
 * @author Alex Merritt, merritt.alex@gatech.edu
 * Re-wrote to use lists instead of arrays.
 */

#ifndef _FAT_CUBIN_INFO_H
#define _FAT_CUBIN_INFO_H

// System includes
#include <stdlib.h>

// CUDA includes
#include <__cudaFatFormat.h>

// Project includes
#include <debug.h>
#include <util/list.h>

/*-------------------------------------- DATA STRUCTURES ---------------------*/

/**
 * Organizes important information about each fat cubin binary a CUDA application
 * may register with the NVIDIA CUDA Runtime. In below comments referred to as
 * a 'cubin blob'. This file merely aims to offer a storage medium for the cubin
 * structures. Allocation, deallocation, serialization and unserialization are
 * not in the scope of this file.
 *
 * The initial sequence of __cudaRegister*() calls produce state that is copied
 * into instances of the below struture. Each __cudaRegisterFatBinary invocation
 * triggers an instance of this structure; subsequent calls such as
 * __cudaRegister{Function|Var|Texture} add to the immediately preceeding fat
 * binary registration.
 */
struct cuda_fatcubin_info
{
	//! List element to link with other fatcubin_info_t
	struct list_head link;

	//! Return value of __cudaRegisterFatBinary used to lookup this struct.
	void **handle;

	//! The data structure associated with the handle. It is the argument to
	//! __cudaRegisterFatBinary.
	__cudaFatCudaBinary *cubin;

	//! List of reg_var_args_t
	struct list_head variable_list;
	int num_vars;

	//! List of reg_tex_args_t
	struct list_head texture_list;
	int num_tex;

	//! List of shared variable names (char*)
	struct list_head shared_var_list;
	int num_shared;

	//! List of reg_func_args_t
	struct list_head function_list;
	int num_funcs;
};

//! A structure to store multiple cubin blobs.
struct fatcubins
{
	struct list_head list; //! List of struct cuda_fat_cubin
	int num_cubins; //! List size
};

/*------------------------------------ INTERNAL/EXTERNAL MACROS --------------*/

/**
 * For-loop to iterate over each cubin blob in a set.
 * @param cubins	Pointer to fatcubins structure.
 * @param entry	Pointer to cuda_fatcubin_info, used as a loop cursor
 */
#define cubins_for_each_cubin(cubins, entry)				\
	list_for_each_entry(entry, &(cubins)->list, link)

/**
 * For-loop to iterate over each variable within a cubin blob.
 * @param entry	Pointer to a reg_var_args_t; loop cursor
 */
#define cubin_for_each_variable(cuda_cubin, entry)		\
	list_for_each_entry(entry, &(cuda_cubin)->variable_list, link)

/**
 * For-loop to iterate over each function within a cubin blob.
 * @param entry	Pointer to a reg_func_args_t; loop cursor
 */
#define cubin_for_each_function(cuda_cubin, entry)		\
	list_for_each_entry(entry, &(cuda_cubin)->function_list, link)

/**
 * For-loop to iterate over each texture within a cubin blob.
 * @param entry	Pointer to a reg_tex_args_t; loop cursor
 */
#define cubin_for_each_texture(cuda_cubin, entry)		\
	list_for_each_entry(entry, &(cuda_cubin)->texture_list, link)

/*-------------------------------------- INTERNAL OPERATIONS -----------------*/

// NOTE: Underscore functions are not meant to be invoked by anyone other than
// the functions in this header file.

static inline void
__fatcubin_init(struct cuda_fatcubin_info *new_cubin_info,
		__cudaFatCudaBinary *cuda_cubin, void **handle)
{
	INIT_LIST_HEAD(&new_cubin_info->link);
	INIT_LIST_HEAD(&new_cubin_info->variable_list);
	INIT_LIST_HEAD(&new_cubin_info->texture_list);
	INIT_LIST_HEAD(&new_cubin_info->shared_var_list);
	INIT_LIST_HEAD(&new_cubin_info->function_list);
	new_cubin_info->num_vars = 0;
	new_cubin_info->num_tex = 0;
	new_cubin_info->num_shared = 0;
	new_cubin_info->num_funcs = 0;
	new_cubin_info->handle = handle;
	new_cubin_info->cubin = cuda_cubin;
}

/*-------------------------------------- EXTERNAL OPERATIONS -----------------*/

/**
 * Locate a cubin blob in the set given the handle.
 */
static inline struct cuda_fatcubin_info *
find_fatcubin_info(struct fatcubins *cubins, void **handle)
{
	struct cuda_fatcubin_info *cuda_cubin;
	if (!cubins) {
		printd(DBG_ERROR, "NULL cubins\n");
		return NULL;
	}
	cubins_for_each_cubin(cubins, cuda_cubin)
		if (handle == cuda_cubin->handle)
			return cuda_cubin;
	return NULL;
}

/**
 * Initialize a fat cubin set.
 */
static inline void
cubins_init(struct fatcubins *cubins)
{
	if (!cubins) {
		printd(DBG_ERROR, "NULL cubins\n");
		return;
	}
	INIT_LIST_HEAD(&cubins->list);
	cubins->num_cubins = 0;
}

// FIXME Add cubins_dealloc to remove all cubin descriptors and registered
// functions, variables, textures, etc

/**
 * Add a CUDA cubin descriptor to the set. This function should be called in
 * tandem with __cudaRegisterFatBinary.
 */
static inline int
cubins_add_cubin(struct fatcubins *cubins,
		__cudaFatCudaBinary *cuda_cubin, void **handle)
{
	struct cuda_fatcubin_info *cubin_info = calloc(1, sizeof(*cubin_info));
	if (!cubin_info) {
		printd(DBG_ERROR, "NULL arg\n");
		return -1;
	}
	__fatcubin_init(cubin_info, cuda_cubin, handle);
	list_add(&cubin_info->link, &cubins->list);
	cubins->num_cubins++;
	return 0;
}

/**
 * Remove a cubin descriptor from the set. This function should be called in
 * tandem with __cudaUnregisterFatBinary.
 */
static inline void
cubins_rm_cubin(struct fatcubins *cubins, void **handle)
{
	struct cuda_fatcubin_info *cuda_cubin;
	if (!cubins) {
		printd(DBG_ERROR, "NULL arg\n");
		return;
	}
	cuda_cubin = find_fatcubin_info(cubins, handle);
	if (!cuda_cubin) {
		printd(DBG_ERROR, "NULL cuda_cubin\n");
		return;
	}
	list_del(&cuda_cubin->link);
	free(cuda_cubin);
	cubins->num_cubins--;
}

/**
 * Add a function associated with the given, previously registered handle to the
 * set. This function should be called in tandem with __cudaRegisterFunction.
 */
static inline void
cubins_add_function(struct fatcubins *cubins,
		void **handle, struct list_head *link)
{
	struct cuda_fatcubin_info *cuda_cubin;
	if (!cubins || !link) {
		printd(DBG_ERROR, "NULL arg\n");
		return;
	}
	cuda_cubin = find_fatcubin_info(cubins, handle);
	if (!cuda_cubin) {
		printd(DBG_ERROR, "NULL cuda_cubin\n");
		return;
	}
	list_add(link, &cuda_cubin->function_list);
	cuda_cubin->num_funcs++;
}

/**
 * Add a variable associated with the given, previously registered handle to the
 * set. This function should be called in tandem with __cudaRegisterVariable.
 */
static inline void
cubins_add_variable(struct fatcubins *cubins,
		void **handle, struct list_head *link)
{
	struct cuda_fatcubin_info *cuda_cubin;
	if (!cubins || !link) {
		printd(DBG_ERROR, "NULL arg\n");
		return;
	}
	cuda_cubin = find_fatcubin_info(cubins, handle);
	if (!cuda_cubin) {
		printd(DBG_ERROR, "NULL cuda_cubin\n");
		return;
	}
	list_add(link, &cuda_cubin->variable_list);
	cuda_cubin->num_vars++;
}

/**
 * Add a texture associated with the given, previously registered handle to the
 * set. This function should be called in tandem with __cudaRegisterTexture.
 */
static inline void
cubins_add_texture(struct fatcubins *cubins,
		void **handle, struct list_head *link)
{
	struct cuda_fatcubin_info *cuda_cubin;
	if (!cubins || !link) {
		printd(DBG_ERROR, "NULL arg\n");
		return;
	}
	cuda_cubin = find_fatcubin_info(cubins, handle);
	if (!cuda_cubin) {
		printd(DBG_ERROR, "NULL cuda_cubin\n");
		return;
	}
	list_add(link, &cuda_cubin->texture_list);
	cuda_cubin->num_tex++;
}

#endif /* _FAT_CUBIN_INFO_H */
