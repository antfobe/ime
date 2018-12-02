#include <linux/err.h>
#include <linux/bio.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/mm.h>
#include <linux/pagemap.h>
#include <linux/refcount.h>
#include <linux/slab.h>
#include <linux/sched.h>
#include "compression.h"

#include <linux/string.h>
#include <linux/cpu.h>
#include <linux/highmem.h>

#include "zcomp.h"

#define LZ4_BTRFS_MAX_WINDOWLOG 17
#define LZ4_BTRFS_MAX_INPUT (1 << LZ4_BTRFS_MAX_WINDOWLOG)
#define LZ4_BTRFS_DEFAULT_LEVEL 3
/*
static LZ4_parameters lz4_get_btrfs_parameters(size_t src_len)
{
	LZ4_parameters params = LZ4_getParams(LZ4_BTRFS_DEFAULT_LEVEL,
						src_len, 0);

	if (params.cParams.windowLog > LZ4_BTRFS_MAX_WINDOWLOG)
		params.cParams.windowLog = LZ4_BTRFS_MAX_WINDOWLOG;
	WARN_ON(src_len > LZ4_BTRFS_MAX_INPUT);
	return params;
}
*/
struct workspace {
	void *mem;
	size_t size;
	char *buf;
	struct list_head list;
	LZ4_inBuffer in_buf;
	LZ4_outBuffer out_buf;
};

static void lz4_free_workspace(struct list_head *ws)
{
	struct workspace *workspace = list_entry(ws, struct workspace, list);

	kvfree(workspace->mem);
	kfree(workspace->buf);
	kfree(workspace);
}

/*** still need to deal with parameters 
 * (feature not implemented in crypto api)***/
static struct list_head *lz4_alloc_workspace(void)
{
    int ret;
    struct zcomp *comp;

	ret = cpuhp_setup_state_multi(CPUHP_ZCOMP_PREPARE, "btrfs_comp:prepare",
				      zcomp_cpu_up_prepare, zcomp_cpu_dead);
	if (ret < 0){
		pr_debug("BTRFS: CPU prepare returned %d\n",
                ret);
        return ERR_PTR(-EBUSY);
    }
    /* We will go with zstd for now */
    comp = zcomp_create("zstd");
	
	struct workspace *workspace;

	workspace = kzalloc(sizeof(*workspace), GFP_KERNEL);
	if (!workspace)
		return ERR_PTR(-ENOMEM);

	workspace->size = PAGE_SIZE * 2;
	workspace->mem = kvmalloc(workspace->size, GFP_KERNEL);
	workspace->buf = kmalloc(PAGE_SIZE, GFP_KERNEL);
	if (!workspace->mem || !workspace->buf)
		goto fail;

	INIT_LIST_HEAD(&workspace->list);

	return &workspace->list;
fail:
	lz4_free_workspace(&workspace->list);
	return ERR_PTR(-ENOMEM);
}

static int lz4_compress_pages(struct list_head *ws,
		struct address_space *mapping,
		u64 start,
		struct page **pages,
		unsigned long *out_pages,
		unsigned long *total_in,
		unsigned long *total_out)
{
	struct workspace *workspace = list_entry(ws, struct workspace, list);
    struct zcomp_strm *stream;
	
    int ret = 0;
	int nr_pages = 0;
	struct page *in_page = NULL;  /* The current page to read */
	struct page *out_page = NULL; /* The current page to write to */
	unsigned long tot_in = 0;
	unsigned long tot_out = 0;
	unsigned long len = *total_out;
	const unsigned long nr_dest_pages = *out_pages;
	unsigned long max_out = nr_dest_pages * PAGE_SIZE;
	LZ4_parameters params = lz4_get_btrfs_parameters(len);

	*out_pages = 0;
	*total_out = 0;
	*total_in = 0;

	/* Initialize the stream */
	stream = zcomp_stream_get(comp);
    if (!stream) {
		pr_warn("BTRFS: compression stream initialization failed\n");
		ret = -EIO;
		goto out;
	}

	/* map in the first page of input data */
	in_page = find_get_page(mapping, start >> PAGE_SHIFT);
	workspace->in_buf.src = kmap(in_page);
	workspace->in_buf.pos = 0;
	workspace->in_buf.size = min_t(size_t, len, PAGE_SIZE);


	/* Allocate and map in the output buffer */
	out_page = alloc_page(GFP_NOFS | __GFP_HIGHMEM);
	if (out_page == NULL) {
		ret = -ENOMEM;
		goto out;
	}
	pages[nr_pages++] = out_page;
	workspace->out_buf.dst = kmap(out_page);
	workspace->out_buf.pos = 0;
	workspace->out_buf.size = min_t(size_t, max_out, PAGE_SIZE);

	while (1) {
		ret = zcomp_compress(stream, workspace->in_buf.src,
				&workspace->in_buf.size);
		if (unlikely(ret)) {
            zcomp_stream_put(comp);
			pr_debug("BTRFS: compression returned %d\n",
					ret);
			ret = -EIO;
			goto out;
		}

		/* Check to see if we are making it bigger */
		if (tot_in + workspace->in_buf.pos > 8192 &&
				tot_in + workspace->in_buf.pos <
				tot_out + workspace->out_buf.pos) {
			ret = -E2BIG;
			goto out;
		}

		/* We've reached the end of our output range */
		if (workspace->out_buf.pos >= max_out) {
			tot_out += workspace->out_buf.pos;
			ret = -E2BIG;
			goto out;
		}

		/* Check if we need more output space */
		if (workspace->out_buf.pos == workspace->out_buf.size) {
			tot_out += PAGE_SIZE;
			max_out -= PAGE_SIZE;
			kunmap(out_page);
			if (nr_pages == nr_dest_pages) {
				out_page = NULL;
				ret = -E2BIG;
				goto out;
			}
			out_page = alloc_page(GFP_NOFS | __GFP_HIGHMEM);
			if (out_page == NULL) {
				ret = -ENOMEM;
				goto out;
			}
			pages[nr_pages++] = out_page;
			workspace->out_buf.dst = kmap(out_page);
			workspace->out_buf.pos = 0;
			workspace->out_buf.size = min_t(size_t, max_out,
							PAGE_SIZE);
		}

		/* We've reached the end of the input */
		if (workspace->in_buf.pos >= len) {
			tot_in += workspace->in_buf.pos;
			break;
		}

		/* Check if we need more input */
		if (workspace->in_buf.pos == workspace->in_buf.size) {
			tot_in += PAGE_SIZE;
			kunmap(in_page);
			put_page(in_page);

			start += PAGE_SIZE;
			len -= PAGE_SIZE;
			in_page = find_get_page(mapping, start >> PAGE_SHIFT);
			workspace->in_buf.src = kmap(in_page);
			workspace->in_buf.pos = 0;
			workspace->in_buf.size = min_t(size_t, len, PAGE_SIZE);
		}
        zcomp_stream_put(comp);
	}
    zcomp_stream_put(comp);
	while (1) {
		if (ret == 0) {
			tot_out += workspace->out_buf.pos;
			break;
		}

		tot_out += PAGE_SIZE;
		max_out -= PAGE_SIZE;
		kunmap(out_page);
		if (nr_pages == nr_dest_pages) {
			out_page = NULL;
			ret = -E2BIG;
			goto out;
		}
		out_page = alloc_page(GFP_NOFS | __GFP_HIGHMEM);
		if (out_page == NULL) {
			ret = -ENOMEM;
			goto out;
		}
		pages[nr_pages++] = out_page;
		workspace->out_buf.dst = kmap(out_page);
		workspace->out_buf.pos = 0;
		workspace->out_buf.size = min_t(size_t, max_out, PAGE_SIZE);
	}

	if (tot_out >= tot_in) {
		ret = -E2BIG;
		goto out;
	}

	ret = 0;
	*total_in = tot_in;
	*total_out = tot_out;
out:
	*out_pages = nr_pages;
	/* Cleanup */
	if (in_page) {
		kunmap(in_page);
		put_page(in_page);
	}
	if (out_page)
		kunmap(out_page);
	return ret;
}

static int lz4_decompress_bio(struct list_head *ws, struct compressed_bio *cb)
{
    struct zcomp_strm *stream;
	
    struct workspace *workspace = list_entry(ws, struct workspace, list);
	struct page **pages_in = cb->compressed_pages;
	u64 disk_start = cb->start;
	struct bio *orig_bio = cb->orig_bio;
	size_t srclen = cb->compressed_len;
	int ret = 0;
	unsigned long page_in_index = 0;
	unsigned long total_pages_in = DIV_ROUND_UP(srclen, PAGE_SIZE);
	unsigned long buf_start;
	unsigned long total_out = 0;

	stream = zcomp_stream_get(comp);
	if (!stream) {
		pr_debug("BTRFS: decompression stream initialization failed\n");
		ret = -EIO;
		goto done;
	}

	workspace->in_buf.src = kmap(pages_in[page_in_index]);
	workspace->in_buf.pos = 0;
	workspace->in_buf.size = min_t(size_t, srclen, PAGE_SIZE);

	workspace->out_buf.dst = workspace->buf;
	workspace->out_buf.pos = 0;
	workspace->out_buf.size = PAGE_SIZE;

	while (1) {
		ret = zcomp_decompress(stream, workspace->in_buf.src,
				workspace->in_buf.size, workspace->out_buf.dst);
		if (unlikely(ret)) {
            zcomp_stream_put(comp);
			pr_debug("BTRFS: stream decompression returned %d\n",
					ret);
			ret = -EIO;
			goto done;
		}
		buf_start = total_out;
		total_out += workspace->out_buf.pos;
		workspace->out_buf.pos = 0;

        size_t endf_ret;
		endf_ret = btrfs_decompress_buf2page(workspace->out_buf.dst,
				buf_start, total_out, disk_start, orig_bio);
		if (ret == 0)
			break;

		if (workspace->in_buf.pos >= srclen)
			break;

		/* Check if we've hit the end of a frame */
		if (endf_ret == 0)
			break;

		if (workspace->in_buf.pos == workspace->in_buf.size) {
			kunmap(pages_in[page_in_index++]);
			if (page_in_index >= total_pages_in) {
				workspace->in_buf.src = NULL;
				ret = -EIO;
				goto done;
			}
			srclen -= PAGE_SIZE;
			workspace->in_buf.src = kmap(pages_in[page_in_index]);
			workspace->in_buf.pos = 0;
			workspace->in_buf.size = min_t(size_t, srclen, PAGE_SIZE);
		}
	}
	ret = 0;
	zero_fill_bio(orig_bio);
done:
	if (workspace->in_buf.src)
		kunmap(pages_in[page_in_index]);
	return ret;
}

static int lz4_decompress(struct list_head *ws, unsigned char *data_in,
		struct page *dest_page,
		unsigned long start_byte,
		size_t srclen, size_t destlen)
{
    struct zcomp_strm *stream;
	
    struct workspace *workspace = list_entry(ws, struct workspace, list);
	int ret = 0;
	unsigned long total_out = 0;
	unsigned long pg_offset = 0;
	char *kaddr;

    stream = zcomp_stream_get(comp);
	if (!stream) {
		pr_warn("BTRFS: decompression stream initialization failed\n");
		ret = -EIO;
		goto finish;
	}

	destlen = min_t(size_t, destlen, PAGE_SIZE);

	workspace->in_buf.src = data_in;
	workspace->in_buf.pos = 0;
	workspace->in_buf.size = srclen;

	workspace->out_buf.dst = workspace->buf;
	workspace->out_buf.pos = 0;
	workspace->out_buf.size = PAGE_SIZE;

    ret = 1;
	while (pg_offset < destlen
	       && workspace->in_buf.pos < workspace->in_buf.size) {
		unsigned long buf_start;
		unsigned long buf_offset;
		unsigned long bytes;

		/* Check if the frame is over and we still need more input */
		if (ret == 0) {
			pr_debug("BTRFS: decompress stream ended early\n");
			ret = -EIO;
			goto finish;
		}
		ret = zcomp_decompress(stream, workspace->in_buf.src,
				workspace->in_buf.size, workspace->out_buf.dst);
		if (unlikely(ret)) {
			pr_debug("BTRFS: stream decompression returned %d\n",
					ret);
			ret = -EIO;
			goto finish;
		}

		buf_start = total_out;
		total_out += workspace->out_buf.pos;
		workspace->out_buf.pos = 0;

		if (total_out <= start_byte)
			continue;

		if (total_out > start_byte && buf_start < start_byte)
			buf_offset = start_byte - buf_start;
		else
			buf_offset = 0;

		bytes = min_t(unsigned long, destlen - pg_offset,
				workspace->out_buf.size - buf_offset);

		kaddr = kmap_atomic(dest_page);
		memcpy(kaddr + pg_offset, workspace->out_buf.dst + buf_offset,
				bytes);
		kunmap_atomic(kaddr);

		pg_offset += bytes;
	}
	ret = 0;
finish:
    zcomp_stream_put(comp);
	if (pg_offset < destlen) {
		kaddr = kmap_atomic(dest_page);
		memset(kaddr + pg_offset, 0, destlen - pg_offset);
		kunmap_atomic(kaddr);
	}
	return ret;
}

///////////////////////////////////////////////////////////////////////////

static int __init init_lz4t(void)
{
    int ret;
    unsigned int comp_len = 0;
    void * buf = decompressed;
    void * dst;
    struct zcomp *comp;
    struct zcomp_strm *strm;
    struct page *page = alloc_page(GFP_NOIO|__GFP_HIGHMEM);
    if(!page) return -ENOMEM;

	ret = cpuhp_setup_state_multi(CPUHP_ZCOMP_PREPARE, "lz4_test:prepare",
				      zcomp_cpu_up_prepare, zcomp_cpu_dead);
	if (ret < 0){
	    printk(KERN_ERR "[lz4t]: CPU Preparation returned < 0 :%d\n", ret);
        return ret;
    }
    /* We will go with lz4hc for now */
    comp = zcomp_create("lz4hc");
    if(!comp) {
	    printk(KERN_ERR "[lz4t]: Could not create zcomp instance, aborting\n");
        return -EINVAL;
    }
    strm = zcomp_stream_get(comp);
    if(!strm) {
	    printk(KERN_ERR "[lz4t]: Could not create strm instance, aborting\n");
        return -EINVAL;
    }
    ret = zcomp_compress(strm, buf, &comp_len);
    zcomp_stream_put(comp);
    if(!ret) {
	    printk(KERN_INFO "[lz4t]: Compressed size: %d\n", comp_len);
    }
	printk(KERN_INFO "[lz4t]: done compression\n");
    strm = zcomp_stream_get(comp);
    dst = kmap_atomic(page);
    ret = zcomp_decompress(strm, strm->buffer, comp_len, dst);
    zcomp_stream_put(comp);
    if(!ret) {
	    printk(KERN_INFO "[lz4t]: Decompressed size: %ld\n", strlen((char *)dst));
	    printk(KERN_INFO "[lz4t]: Decompressed string: %s\n", (char *)dst);
    }
    kunmap_atomic(dst);
    __free_page(page);
    zcomp_destroy(comp);
	return 0;
}

static void __exit exit_lz4t(void)
{
    cpuhp_remove_multi_state(CPUHP_ZCOMP_PREPARE);
	printk(KERN_INFO "[lz4t]: Goodbye, world\n");
}

module_init(init_lz4t);
module_exit(exit_lz4t);
