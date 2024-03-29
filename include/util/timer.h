/**
 * @file timer.h
 * @brief Header for time measurement etc
 *
 * @author Romain Cledat, romain@cc.gatech.edu
 * @author Vishakha Gupta, vishakha@cc.gatech.edu
 * @author Alex Merritt, merritt.alex@gatech.edu (code refactoring)
 * @bug There might be a wastage of space
 *
 * TODO Need an example call trace for these functions. Or an explanation.
 */
#ifndef TIMER_H_
#define TIMER_H_

// System includes
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <stdint.h>
#include <time.h>

/*-------------------------------------- DEFINITIONS -------------------------*/

#define TIMERMSG_PREFIX	"[TIMER] "

#define TIMER_ACCURACY MICROSECONDS

/*-------------------------------------- DATA TYPES --------------------------*/

enum answerUnit_t {
		SECONDS,
		MILLISECONDS,
		MICROSECONDS,
		NANOSECONDS
};

struct timer {
	clockid_t id;
	struct timespec s, e; // start, end
	struct timespec accum;
	bool isPaused;
};

/*-------------------------------------- MACROS ------------------------------*/

#ifdef TIMING

// Use these to declare some number of timers.
#define TIMER_DECLARE1(t1) \
	struct timer t1;
#define TIMER_DECLARE2(t1,t2) \
	struct timer t1,t2;
#define TIMER_DECLARE3(t1,t2,t3) \
	struct timer t1,t2,t3;
#define TIMER_DECLARE4(t1,t2,t3,t4) \
	struct timer t1,t2,t3,t4;

#define TIMER_START(tim) \
	do { \
		timer_init(CLOCK_REALTIME, &tim); \
		timer_start(&tim); \
	} while (0)

#define TIMER_PAUSE(tim)		timer_pause(&tim, TIMER_ACCURACY)
#define TIMER_RESUME(tim)		timer_resume(&tim);
#define TIMER_CLEAR(tim)		timer_reset(&tim)

/** Stop the timer and store the measured value in clk (must be uint64_t). */
#define TIMER_END(tim,clk)	((clk) = timer_end(&tim, TIMER_ACCURACY))

#else	/* !TIMING */

#define TIMER_DECLARE1(t1)
#define TIMER_DECLARE2(t1,t2)
#define TIMER_DECLARE3(t1,t2,t3)
#define TIMER_DECLARE4(t1,t2,t3,t4)
#define TIMER_START(tim)
#define TIMER_PAUSE(tim)
#define TIMER_RESUME(tim)
#define TIMER_CLEAR(tim)
#define TIMER_END(clk,tim)

#endif	/* TIMING */
/*-------------------------------------- FUNCTIONS ---------------------------*/

static inline struct timer *
timer_init_alloc(clockid_t id_)
{ 
	// Default id is CLOCK_REALTIME
	struct timer *tm = (struct timer *)calloc(1, sizeof(struct timer));
	tm->id = id_;
	tm->isPaused = true;

	tm->accum.tv_sec = 0;
	tm->accum.tv_nsec = 0;

	return tm;
}

static inline int
timer_init(clockid_t id_, struct timer *tm)
{ 
	tm->id = id_;
	tm->isPaused = true;

	tm->accum.tv_sec = 0;
	tm->accum.tv_nsec = 0;

	return 0;
}

static inline void
timer_destroy(struct timer *tm) 
{ 
	if (tm != NULL)
		free(tm);
}

static void
timer_start(struct timer *tm)
{
	tm->accum.tv_sec = 0;
	tm->accum.tv_nsec = 0;
	tm->isPaused = false;
	if(clock_gettime(tm->id, &tm->s) != 0) {
		perror("-- ERR -- HRTIMER: error in getting start time");
	}
}

static void
timer_reset(struct timer *tm)
{
	tm->accum.tv_sec = 0;
	tm->accum.tv_nsec = 0;
	tm->isPaused = false;
}

static uint64_t
timer_end(struct timer *tm, enum answerUnit_t unit) 
{
	if(clock_gettime(tm->id, &tm->e) != 0) {
		perror("-- ERR -- HRTimer: error in getting end time");
		return 0;
	}
	if(!tm->isPaused) {
		if((tm->e.tv_sec < tm->s.tv_sec) || 
				(tm->e.tv_sec == tm->s.tv_sec && 
				 	tm->e.tv_nsec < tm->s.tv_nsec)) { 
			fprintf(stderr, "-- WAR -- HRTimer: Elapsed time was negative\n");
		} 
		else {
			tm->accum.tv_sec += tm->e.tv_sec - tm->s.tv_sec;
			if(tm->e.tv_nsec < tm->s.tv_nsec) {
				tm->accum.tv_sec -= 1;
				tm->e.tv_nsec += (long int)1e9;
			}
			tm->accum.tv_nsec += tm->e.tv_nsec - tm->s.tv_nsec;
		}
		while(tm->accum.tv_nsec >= 1e9) {
			tm->accum.tv_sec += 1;
			tm->accum.tv_nsec -= (long int)1e9;
		}
	}
	switch(unit) {
		case SECONDS:
			return (uint64_t)tm->accum.tv_sec;
		case MILLISECONDS:
			return (uint64_t)tm->accum.tv_sec *1000 + (uint64_t)tm->accum.tv_nsec/(uint64_t)1e6;
		case MICROSECONDS:
			return (uint64_t)tm->accum.tv_sec * (uint64_t)1e6 + (uint64_t)tm->accum.tv_nsec/(uint64_t)1e3;
		case NANOSECONDS:
			return (uint64_t)tm->accum.tv_sec * (uint64_t)1e9 + (uint64_t)tm->accum.tv_nsec;
	};
	return 0; // To keep the compiler happy
}

static inline uint64_t
timer_pause(struct timer *tm, enum answerUnit_t unit) 
{
	struct timespec temp ;
	if(clock_gettime(tm->id, &tm->e) != 0) {
		perror("-- ERR -- HRTimer: error in getting end time");
		return 0;
	}

	temp.tv_sec = 0;
	temp.tv_nsec = 0;

	if((tm->e.tv_sec < tm->s.tv_sec) || 
		(tm->e.tv_sec == tm->s.tv_sec && 
		 	tm->e.tv_nsec < tm->s.tv_nsec)) { 
		fprintf(stderr, "-- WAR -- HRTimer: Elapsed time was negative\n");
	} 
	else {
		temp.tv_sec += tm->e.tv_sec - tm->s.tv_sec;
		if(tm->e.tv_nsec < tm->s.tv_nsec) {
			temp.tv_sec -= 1;
			tm->e.tv_nsec += (unsigned long)1e9;
		}
		temp.tv_nsec += tm->e.tv_nsec - tm->s.tv_nsec;
	}

	while(temp.tv_nsec >= 1e9) {
		temp.tv_sec +=1;
		temp.tv_nsec -= (unsigned long)1e9;
	}

	tm->accum.tv_sec += temp.tv_sec;
	tm->accum.tv_nsec += temp.tv_nsec;
	while(tm->accum.tv_nsec >= 1e9) {
		tm->accum.tv_sec += 1;
		tm->accum.tv_nsec -= (unsigned long)1e9;
	}

	tm->isPaused = true;

	switch(unit) {
		case SECONDS:
			return (uint64_t)temp.tv_sec;
		case MILLISECONDS:
			return (uint64_t)temp.tv_sec * 1000 + (uint64_t)temp.tv_nsec/(uint64_t)1e6;
		case MICROSECONDS:
			return (uint64_t)temp.tv_sec * (uint64_t)1e6 + (uint64_t)temp.tv_nsec/(uint64_t)1e3;;
		case NANOSECONDS:
			return (uint64_t)temp.tv_sec * (uint64_t)1e9 + (uint64_t)temp.tv_nsec;
	};
	return 0; // to keep the compiler happy
}

static inline void
timer_resume(struct timer *tm) {
	tm->isPaused = false;
	if(clock_gettime(tm->id, &tm->s) != 0) {
		perror("-- ERR -- HRTimer: error in getting start time");
	}
}

static inline struct timespec
timer_getTimeSpec(struct timer *tm) 
{
	return tm->accum;
}

static inline uint64_t
timer_getTime(struct timer *tm, enum answerUnit_t unit) 
{
	switch(unit) {
		case SECONDS:
			return (uint64_t)tm->accum.tv_sec;
		case MILLISECONDS:
			return (uint64_t)tm->accum.tv_sec *1000 + (uint64_t)tm->accum.tv_nsec/(uint64_t)1e6;
		case MICROSECONDS:
			return (uint64_t)tm->accum.tv_sec * (uint64_t)1e6 + (uint64_t)tm->accum.tv_nsec/(uint64_t)1e3;
		case NANOSECONDS:
			return (uint64_t)tm->accum.tv_sec * (uint64_t)1e9 + (uint64_t)tm->accum.tv_nsec;
	};
	return 0; // To keep the compiler happy
}

#endif				/* TIMER_H_ */
