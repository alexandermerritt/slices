
// TODO define the message format for the queues


#define SHM_KEY_FMT			"/assembly-%s-%s"		// pid-tid
#define SHM_MAX_FMT_LEN		255

#define MQ_PERMS				0660
#define MQ_OPEN_LEADER_FLAGS	(O_RDWR | O_CREAT | O_EXCL)
#define MQ_OPEN_MEMBER_FLAGS	(O_RDWR)
#define MQ_INVALID_VALUE		((mqd_t) - 1) // RTFM
#define MQ_IS_VALID(m)			((m) != MQ_INVALID_VALUE)

#define MEMB_DIR_PREFIX			"/tmp"
#define MEMB_DIR_MAX_LEN		512
#define MEMB_DIR_LEN			(MEMB_DIR_MAX_LEN - SHMGRP_MAX_KEY_LEN)
// FIXME Verify strlen(MEMB_DIR_PREFIX) + SHMGRP_MAX_KEY_LEN <= MEMB_DIR_MAX_LEN
#define MEMB_DIR_PERMS			0660
