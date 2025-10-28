import mysql.connector
import random
from datetime import datetime, timedelta

DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'root',
    'password': '',
    'database': 'mobile_repair'
}

SAMPLES = [
    "Excellent service, fixed my phone quickly and professionally.",
    "Good work but took longer than expected.",
    "Screen still flickers after repair.",
    "Average experience. Price was okay.",
    "Terrible — phone was damaged after service.",
    "Friendly staff and fast turnaround.",
    "They replaced my battery and now it lasts longer.",
    "Parts were a bit expensive but quality was good.",
    "I had to go back twice to finish the job.",
    "Highly recommended for urgent repairs!"
]

def random_past_time():
    # random time within the last 60 days
    return datetime.now() - timedelta(days=random.randint(0, 60), hours=random.randint(0,23))


def main(n=50):
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    for i in range(n):
        name = f"User{i+1}"
        email = f"user{i+1}@example.com"
        rating = random.randint(1, 5)
        review = random.choice(SAMPLES)
        created = random_past_time().strftime('%Y-%m-%d %H:%M:%S')
        cur.execute("""
            INSERT INTO comments (customer_name, email, rating, review_content, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (name, email, rating, review, created, created))
    conn.commit()
    cur.close()
    conn.close()
    print(f"Inserted {n} dummy reviews.")


if __name__ == "__main__":
    main(100)
