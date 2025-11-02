/*
1. 선점형 우선순위 스케줄링: 
우선순위가 높은 스레드가 항상 먼저 실행되도록 ready_list와 세마포어/뮤텍스 등의
동기화 대기열을 우선순위 순으로 유지한다. 실행 중 스레드가 thread_set_priority()로
우선순위를 낮추면 즉시 더 높은 우선순위 스레드가 Preempt 해야 하며, 대기 중이던
높은 우선순위 스레드가 생성되거나 Unblock 될 때에도 즉시 Context Switch가 일어나야 한다.
세마포어 및 조건 변수 대기열 역시 우선순위 순으로 정렬하며 Unblock마다 
최고 우선순위 스레드를 선택한다.

2. 에이징 적용: 낮은 우선순위 스레드의 기아상태(Starvation)를 방지하기 위해
준비 큐에서 대기 중인 스레드의 age 값을 매 틱마다 1씩 증가시키고,
age가 20에 도달하면 우선순위를 한 단계 상승시킨 뒤 age를 0으로 초기화한다.
age 값은 큐에 추가될 때마다 0으로 초기화 된다. 이 과정을 반복해 스레드가 
기본 우선순위(PRI_DEFAULT)까지 회복하고 필요한 시점에 다른 스레드를 선점하도록 한다.

3. Simplified MLFQS(Multi-Level Feedback Queue Scheduler): -mlfqs 플래그 사용 시
세 단계(Q0, Q1, Q2)의 피드백 큐를 사용하는 단순한 형태의 MLFQS를 구현한다.
모든 스레드는 Q0에서 시작하고, 각 큐의 타임 슬라이스는 Q0=2틱, Q1=4틱, Q2=8틱이다.
이 때 Q1 실행 중 Q0에 새 스레드가 추가되면 즉시 새 스레드가 실행된다. 
주어진 시간 슬라이스를 모두 소모하면 Q0→Q1→Q2 순으로 강등되고, 
대기 중에는 에이징 정책과 동일하게 매 틱 age를 1씩 증가시켜 age가 20이 되면
한 단계 상위 큐로 승급시키며 age를 0으로 초기화한다. 이때 상위 큐가 비어 있어야만
다음 큐의 스레드를 실행하며, 큐의 승급과 강등은 타이머 틱 단위로 수행한다.
*/



#include "threads/thread.h"
#include <debug.h>
#include <stddef.h>
#include <random.h>
#include <stdio.h>
#include <string.h>
#include "threads/flags.h"
#include "threads/interrupt.h"
#include "threads/intr-stubs.h"
#include "threads/palloc.h"
#include "threads/switch.h"
#include "threads/synch.h"
#include "threads/vaddr.h"
#ifdef USERPROG
#include "userprog/process.h"
#endif

/* Random value for struct thread's `magic' member.
   Used to detect stack overflow.  See the big comment at the top
   of thread.h for details. */
#define THREAD_MAGIC 0xcd6abf4b

/* List of processes in THREAD_READY state, that is, processes
   that are ready to run but not actually running. */
static struct list ready_list;

/* List of all processes.  Processes are added to this list
   when they are first scheduled and removed when they exit. */
static struct list all_list;

/* List of process in sleep */
static struct list sleep_list;
static int64_t next_tick_to_wakeup = INT64_MAX;

/* Idle thread. */
static struct thread *idle_thread;

/* Initial thread, the thread running init.c:main(). */
static struct thread *initial_thread;

/* Lock used by allocate_tid(). */
static struct lock tid_lock;

/* Stack frame for kernel_thread(). */
struct kernel_thread_frame
{
    void *eip;             /* Return address. */
    thread_func *function; /* Function to call. */
    void *aux;             /* Auxiliary data for function. */
};

/* Statistics. */
static long long idle_ticks;   /* # of timer ticks spent idle. */
static long long kernel_ticks; /* # of timer ticks in kernel threads. */
static long long user_ticks;   /* # of timer ticks in user programs. */

/* Scheduling. */
#define TIME_SLICE 4          /* # of timer ticks to give each thread. */
static unsigned thread_ticks; /* # of timer ticks since last yield. */

/* If false (default), use round-robin scheduler.
   If true, use multi-level feedback queue scheduler.
   Controlled by kernel command-line option "-o mlfqs". */
bool thread_mlfqs;

static void kernel_thread (thread_func *, void *aux);

static void idle (void *aux UNUSED);
static struct thread *running_thread (void);
static struct thread *next_thread_to_run (void);
static void init_thread (struct thread *, const char *name, int priority);
static bool is_thread (struct thread *) UNUSED;
static void *alloc_frame (struct thread *, size_t size);
static void schedule (void);
void thread_schedule_tail (struct thread *prev);
static tid_t allocate_tid (void);

/* Initializes the threading system by transforming the code
   that's currently running into a thread.  This can't work in
   general and it is possible in this case only because loader.S
   was careful to put the bottom of the stack at a page boundary.

   Also initializes the run queue and the tid lock.

   After calling this function, be sure to initialize the page
   allocator before trying to create any threads with
   thread_create().

   It is not safe to call thread_current() until this function
   finishes. */
void
thread_init (void)
{
    ASSERT (intr_get_level () == INTR_OFF);

    lock_init (&tid_lock);
    list_init (&ready_list);
    list_init (&all_list);
    list_init (&sleep_list);

    /* Set up a thread structure for the running thread. */
    initial_thread = running_thread ();
    init_thread (initial_thread, "main", PRI_DEFAULT);
    initial_thread->status = THREAD_RUNNING;
    initial_thread->tid = allocate_tid ();
}

/* Starts preemptive thread scheduling by enabling interrupts.
   Also creates the idle thread. */
void
thread_start (void)
{
    /* Create the idle thread. */
    struct semaphore idle_started;
    sema_init (&idle_started, 0);
    thread_create ("idle", PRI_MIN, idle, &idle_started);

    /* Start preemptive thread scheduling. */
    intr_enable ();

    /* Wait for the idle thread to initialize idle_thread. */
    sema_down (&idle_started);
}

/* Called by the timer interrupt handler at each timer tick.
   Thus, this function runs in an external interrupt context. */
void
thread_tick (void)
{
    struct thread *t = thread_current ();

    /* Update statistics. */
    if (t == idle_thread)
        idle_ticks++;
#ifdef USERPROG
    else if (t->pagedir != NULL)
        user_ticks++;
#endif
    else
        kernel_ticks++;

    /* Enforce preemption. */
    if (++thread_ticks >= TIME_SLICE)
        intr_yield_on_return ();

   agingtoThread ();

   
}

/* Prints thread statistics. */
void
thread_print_stats (void)
{
    printf ("Thread: %lld idle ticks, %lld kernel ticks, %lld user ticks\n",
            idle_ticks, kernel_ticks, user_ticks);
}

/* Creates a new kernel thread named NAME with the given initial
   PRIORITY, which executes FUNCTION passing AUX as the argument,
   and adds it to the ready queue.  Returns the thread identifier
   for the new thread, or TID_ERROR if creation fails.

   If thread_start() has been called, then the new thread may be
   scheduled before thread_create() returns.  It could even exit
   before thread_create() returns.  Contrariwise, the original
   thread may run for any amount of time before the new thread is
   scheduled.  Use a semaphore or some other form of
   synchronization if you need to ensure ordering.

   The code provided sets the new thread's `priority' member to
   PRIORITY, but no actual priority scheduling is implemented.
   Priority scheduling is the goal of Problem 1-3. */
tid_t
thread_create (const char *name, int priority,
               thread_func *function, void *aux)
{
    struct thread *t;
    struct kernel_thread_frame *kf;
    struct switch_entry_frame *ef;
    struct switch_threads_frame *sf;
    tid_t tid;
    enum intr_level old_level;

    ASSERT (function != NULL);

    /* Allocate thread. */
    t = palloc_get_page (PAL_ZERO);
    if (t == NULL)
        return TID_ERROR;

    /* Initialize thread. */
    init_thread (t, name, priority);
    tid = t->tid = allocate_tid ();

    /* Prepare thread for first run by initializing its stack.
     Do this atomically so intermediate values for the 'stack' 
     member cannot be observed. */
    old_level = intr_disable ();

    /* Stack frame for kernel_thread(). */
    kf = alloc_frame (t, sizeof *kf);
    kf->eip = NULL;
    kf->function = function;
    kf->aux = aux;

    /* Stack frame for switch_entry(). */
    ef = alloc_frame (t, sizeof *ef);
    ef->eip = (void (*) (void))kernel_thread;

    /* Stack frame for switch_threads(). */
    sf = alloc_frame (t, sizeof *sf);
    sf->eip = switch_entry;
    sf->ebp = 0;

    intr_set_level (old_level);

    /* Add to run queue. */
    thread_unblock (t);

    return tid;
}

/* Puts the current thread to sleep.  It will not be scheduled
   again until awoken by thread_unblock().

   This function must be called with interrupts turned off.  It
   is usually a better idea to use one of the synchronization
   primitives in synch.h. */
void
thread_block (void)
{
    ASSERT (!intr_context ());
    ASSERT (intr_get_level () == INTR_OFF);

    thread_current ()->status = THREAD_BLOCKED;
    schedule ();
}

/* Transitions a blocked thread T to the ready-to-run state.
   This is an error if T is not blocked.  (Use thread_yield() to
   make the running thread ready.)

   This function does not preempt the running thread.  This can
   be important: if the caller had disabled interrupts itself,
   it may expect that it can atomically unblock a thread and
   update other data. */
void
thread_unblock (struct thread *t)
{
    enum intr_level old_level;

    ASSERT (is_thread (t));

    old_level = intr_disable ();
    ASSERT (t->status == THREAD_BLOCKED);
    // age를 0으로 만들기
    t->age = 0;
    list_insert_ordered (&ready_list, &t->elem, comparePriority, NULL);
   //스레드 t를 삽입을했음 근데 새로 깨어난 스레드의 우선순위가 더 높다면
   // 현재스레드는 CPU를 양보해야한다... 바로 thread_yield()호출
   //
    t->status = THREAD_READY;
    intr_set_level (old_level);
    if (t->priority > thread_current ()->priority){
    thread_yield (); //if문으로 우선순위 비교함
   }
}

/* Returns the name of the running thread. */
const char *
thread_name (void)
{
    return thread_current ()->name;
}

/* Returns the running thread.
   This is running_thread() plus a couple of sanity checks.
   See the big comment at the top of thread.h for details. */
struct thread *
thread_current (void)
{
    struct thread *t = running_thread ();

    /* Make sure T is really a thread.
     If either of these assertions fire, then your thread may
     have overflowed its stack.  Each thread has less than 4 kB
     of stack, so a few big automatic arrays or moderate
     recursion can cause stack overflow. */
    ASSERT (is_thread (t));
    ASSERT (t->status == THREAD_RUNNING);

    return t;
}

/* Returns the running thread's tid. */
tid_t
thread_tid (void)
{
    return thread_current ()->tid;
}

/* Deschedules the current thread and destroys it.  Never
   returns to the caller. */
void
thread_exit (void)
{
    ASSERT (!intr_context ());

#ifdef USERPROG
    process_exit ();
#endif

    /* Remove thread from all threads list, set our status to dying,
     and schedule another process.  That process will destroy us
     when it calls thread_schedule_tail(). */
    intr_disable ();
    list_remove (&thread_current ()->allelem);
    thread_current ()->status = THREAD_DYING;
    schedule ();
    NOT_REACHED ();
}

/* Yields the CPU.  The current thread is not put to sleep and
   may be scheduled again immediately at the scheduler's whim. */
void
thread_yield (void)
{
    struct thread *cur = thread_current ();
    enum intr_level old_level;

    ASSERT (!intr_context ());

    old_level = intr_disable ();
   //ready_list에 우선순위대로 삽입한다.list_insert_ordered()
    if (cur != idle_thread){
        cur->age = 0; //0으로 바꿔주기만 하면 된다.
        list_insert_ordered (&ready_list, &cur->elem, comparePriority, NULL);
    }
    cur->status = THREAD_READY;
    schedule ();
    intr_set_level (old_level);
}

/* Invoke function 'func' on all threads, passing along 'aux'.
   This function must be called with interrupts off. */
void
thread_foreach (thread_action_func *func, void *aux)
{
    struct list_elem *e;

    ASSERT (intr_get_level () == INTR_OFF);

    for (e = list_begin (&all_list); e != list_end (&all_list);
         e = list_next (e))
        {
            struct thread *t = list_entry (e, struct thread, allelem);
            func (t, aux);
        }
}

/*1번 우선순위 스케줄링 부분*/
//comparePriority는 우선순위를 비교한다.. 
bool
comparePriority (const struct list_elem *a,
                  const struct list_elem *b,
                  void *aux UNUSED)
{
    struct thread *thread_a = list_entry (a, struct thread, elem);
    struct thread *thread_b = list_entry (b, struct thread, elem);

    return thread_a->priority > thread_b->priority;
}


/* Sets the current thread's priority to NEW_PRIORITY. */
void
thread_set_priority (int new_priority)
{
    thread_current ()->priority = new_priority;
   if (!list_empty (&ready_list) &&
    (new_priority < list_entry (list_begin (&ready_list), struct thread, elem)->priority)){
    thread_yield ();
}
   
}

/* Returns the current thread's priority. */
int
thread_get_priority (void)
{
    return thread_current ()->priority;
}

/* Sets the current thread's nice value to NICE. */
void
thread_set_nice (int nice UNUSED)
{
    /* Not yet implemented. */
}

/* Returns the current thread's nice value. */
int
thread_get_nice (void)
{
    /* Not yet implemented. */
    return 0;
}

/* Returns 100 times the system load average. */
int
thread_get_load_avg (void)
{
    /* Not yet implemented. */
    return 0;
}

/* Returns 100 times the current thread's recent_cpu value. */
int
thread_get_recent_cpu (void)
{
    /* Not yet implemented. */
    return 0;
}

/* Idle thread.  Executes when no other thread is ready to run.

   The idle thread is initially put on the ready list by
   thread_start().  It will be scheduled once initially, at which
   point it initializes idle_thread, "up"s the semaphore passed
   to it to enable thread_start() to continue, and immediately
   blocks.  After that, the idle thread never appears in the
   ready list.  It is returned by next_thread_to_run() as a
   special case when the ready list is empty. */
static void
idle (void *idle_started_ UNUSED)
{
    struct semaphore *idle_started = idle_started_;
    idle_thread = thread_current ();
    sema_up (idle_started);

    for (;;)
        {
            /* Let someone else run. */
            intr_disable ();
            thread_block ();

            /* Re-enable interrupts and wait for the next one.

         The `sti' instruction disables interrupts until the
         completion of the next instruction, so these two
         instructions are executed atomically.  This atomicity is
         important; otherwise, an interrupt could be handled
         between re-enabling interrupts and waiting for the next
         one to occur, wasting as much as one clock tick worth of
         time.

         See [IA32-v2a] "HLT", [IA32-v2b] "STI", and [IA32-v3a]
         7.11.1 "HLT Instruction". */
            asm volatile ("sti; hlt" : : : "memory");
        }
}


/*thread_tick에서 호출하는 에이징 함수!! */
static void
agingtoThread (void){
    struct list_elem *e = list_begin (&ready_list);
   
    while (e != list_end (&ready_list)){
       // 리스트의 끝까지 루프를 돈다(list_end에 도달할때까지)
        struct thread *t = list_entry (e, struct thread, elem);
       // 검사하다가 age가 20을 넘는 스레드라서 재정렬을 해버리면
       // 꼬여버리니까 다음에 검사해야 할 걸 미리 저장해둔다
        struct list_elem *next_e = list_next(e); 
       // 모든 스레드에게 age++ 해준다.
        t->age++; 
       // age가 20이 넘는지 검사한다
        if (t->age >= 20 && t->priority < PRI_MAX){
           // (스레드t의age 값이 20살 이상이고&&우선순위의 최대치가 아니라면)
           // 우선순위가 최대면 어차피 맨 앞에 있을 것... 
            t->priority++; // 우선순위 1 상승시키고
            t->age = 0;    // 0으로 리셋
            
            // 이 경우 이제 우선순위가 바뀌었으니 리스트를 재정렬해줘야 한다.
            list_remove (&t->elem);// 리스트에서 뺐다가 
            list_insert_ordered (&ready_list, &t->elem, comparePriority, NULL);
           // 다시 넣는다
        }// 반복문종료...
        
        // 이러면 리스트순서가바뀌엇을수도있고안바뀌었을수도 있다
       //하지만 어차피 다음에 검사해야 할 스레드는 앞에서 미리 정해놨기 때문에
       // 순서가 바뀌든 말든 문제 없이 넘어갈 수 있다.
        e = next_e; 
    }
}





/* Function used as the basis for a kernel thread. */
static void
kernel_thread (thread_func *function, void *aux)
{
    ASSERT (function != NULL);

    intr_enable (); /* The scheduler runs with interrupts off. */
    function (aux); /* Execute the thread function. */
    thread_exit (); /* If function() returns, kill the thread. */
}

/* Returns the running thread. */
struct thread *
running_thread (void)
{
    uint32_t *esp;

    /* Copy the CPU's stack pointer into `esp', and then round that
     down to the start of a page.  Because `struct thread' is
     always at the beginning of a page and the stack pointer is
     somewhere in the middle, this locates the curent thread. */
    asm ("mov %%esp, %0" : "=g"(esp));
    return pg_round_down (esp);
}

/* Returns true if T appears to point to a valid thread. */
static bool
is_thread (struct thread *t)
{
    return t != NULL && t->magic == THREAD_MAGIC;
}

/* Does basic initialization of T as a blocked thread named
   NAME. */
static void
init_thread (struct thread *t, const char *name, int priority)
{
    ASSERT (t != NULL);
    ASSERT (PRI_MIN <= priority && priority <= PRI_MAX);
    ASSERT (name != NULL);

    memset (t, 0, sizeof *t);
    t->status = THREAD_BLOCKED;
    strlcpy (t->name, name, sizeof t->name);
    t->stack = (uint8_t *)t + PGSIZE;
    t->priority = priority;
    t->magic = THREAD_MAGIC;
   //다른 변수들 초기화 할 때 같이 0으로 초기화해준다 
    t->age = 0;
    list_push_back (&all_list, &t->allelem);
}

/* Allocates a SIZE-byte frame at the top of thread T's stack and
   returns a pointer to the frame's base. */
static void *
alloc_frame (struct thread *t, size_t size)
{
    /* Stack data is always allocated in word-size units. */
    ASSERT (is_thread (t));
    ASSERT (size % sizeof (uint32_t) == 0);

    t->stack -= size;
    return t->stack;
}

/* Chooses and returns the next thread to be scheduled.  Should
   return a thread from the run queue, unless the run queue is
   empty.  (If the running thread can continue running, then it
   will be in the run queue.)  If the run queue is empty, return
   idle_thread. */
static struct thread *
next_thread_to_run (void)
{
    if (list_empty (&ready_list))
        return idle_thread;
    else
        return list_entry (list_pop_front (&ready_list), struct thread, elem);
}

/* Completes a thread switch by activating the new thread's page
   tables, and, if the previous thread is dying, destroying it.

   At this function's invocation, we just switched from thread
   PREV, the new thread is already running, and interrupts are
   still disabled.  This function is normally invoked by
   thread_schedule() as its final action before returning, but
   the first time a thread is scheduled it is called by
   switch_entry() (see switch.S).

   It's not safe to call printf() until the thread switch is
   complete.  In practice that means that printf()s should be
   added at the end of the function.

   After this function and its caller returns, the thread switch
   is complete. */
void
thread_schedule_tail (struct thread *prev)
{
    struct thread *cur = running_thread ();

    ASSERT (intr_get_level () == INTR_OFF);

    /* Mark us as running. */
    cur->status = THREAD_RUNNING;

    /* Start new time slice. */
    thread_ticks = 0;

#ifdef USERPROG
    /* Activate the new address space. */
    process_activate ();
#endif

    /* If the thread we switched from is dying, destroy its struct
     thread.  This must happen late so that thread_exit() doesn't
     pull out the rug under itself.  (We don't free
     initial_thread because its memory was not obtained via
     palloc().) */
    if (prev != NULL && prev->status == THREAD_DYING && prev != initial_thread)
        {
            ASSERT (prev != cur);
            palloc_free_page (prev);
        }
}

/* Schedules a new process.  At entry, interrupts must be off and
   the running process's state must have been changed from
   running to some other state.  This function finds another
   thread to run and switches to it.

   It's not safe to call printf() until thread_schedule_tail()
   has completed. */
static void
schedule (void)
{
    struct thread *cur = running_thread ();
    struct thread *next = next_thread_to_run ();
    struct thread *prev = NULL;

    ASSERT (intr_get_level () == INTR_OFF);
    ASSERT (cur->status != THREAD_RUNNING);
    ASSERT (is_thread (next));

    if (cur != next)
        prev = switch_threads (cur, next);
    thread_schedule_tail (prev);
}

/* Returns a tid to use for a new thread. */
static tid_t
allocate_tid (void)
{
    static tid_t next_tid = 1;
    tid_t tid;

    lock_acquire (&tid_lock);
    tid = next_tid++;
    lock_release (&tid_lock);

    return tid;
}

/* Offset of `stack' member within `struct thread'.
   Used by switch.S, which can't figure it out on its own. */
uint32_t thread_stack_ofs = offsetof (struct thread, stack);
