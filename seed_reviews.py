"""
Seed realistic customer reviews for BingleFixIt phone/computer repair and IT support business.
This populates the 'comments' table with diverse, realistic customer feedback.
"""

import mysql.connector
import random
from datetime import datetime, timedelta
from typing import List, Tuple

class ReviewSeeder:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'port': 3307,
            'user': 'root',
            'password': 'frince101',
            'database': 'mobile_repair'
        }
        
        # Realistic customer names
        self.names = [
            "Sarah Johnson", "Mike Chen", "Emily Rodriguez", "James Williams",
            "Ashley Thompson", "David Kim", "Jessica Martinez", "Chris Anderson",
            "Amanda Lee", "Robert Taylor", "Maria Garcia", "Kevin Brown",
            "Laura Davis", "Ryan Wilson", "Nicole Moore", "Daniel Jackson",
            "Michelle White", "Brian Harris", "Jennifer Clark", "Steven Lewis"
        ]
        
        # Email patterns
        self.email_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com"]
        
        # Review templates by rating (5-star to 1-star)
        self.review_templates = {
            5: [
                "Excellent service! My {device} {issue} was fixed perfectly. The technician was professional and fast. Highly recommend!",
                "Outstanding work! {name} fixed my {device} {issue} in no time. Great customer service and fair pricing. Will definitely return!",
                "Amazing experience! They repaired my {device} {issue} same day. Very knowledgeable staff and quality parts used.",
                "Perfect service from start to finish. My {device} {issue} was resolved quickly. The warranty gave me peace of mind.",
                "Fantastic repair shop! Fixed my {device} {issue} professionally. Friendly staff and reasonable prices. Five stars!",
                "Best tech repair in town! My {device} {issue} was handled expertly. Fast turnaround and the device works like new.",
                "Wonderful experience! They diagnosed and fixed my {device} {issue} efficiently. Very satisfied with the quality of work.",
                "Highly professional! My {device} {issue} was repaired perfectly. Great communication throughout the process."
            ],
            4: [
                "Good service overall. My {device} {issue} was fixed properly. Took a bit longer than expected but quality work.",
                "Solid repair job on my {device} {issue}. Professional technicians and fair pricing. Would use again.",
                "Pretty satisfied with the service. My {device} {issue} is working well now. Friendly staff and clean shop.",
                "Good experience. They fixed my {device} {issue} correctly. Pricing was reasonable and staff was helpful.",
                "Reliable service for my {device} {issue}. The repair was done well, though I had to wait a few days.",
                "Nice work on my {device} {issue}. Professional and efficient. Just wish it was a bit faster.",
                "Pleased with the repair. My {device} {issue} was resolved and the technician explained everything clearly.",
                "Good quality repair for my {device} {issue}. Staff was knowledgeable and prices were competitive."
            ],
            3: [
                "Average experience. My {device} {issue} was fixed but took longer than quoted. Service was okay.",
                "Decent service. They resolved my {device} {issue} but communication could be better. Acceptable overall.",
                "Mixed feelings. My {device} {issue} works now but the wait time was longer than expected. Okay pricing.",
                "Service was fine. Fixed my {device} {issue} but nothing exceptional. Would consider other options next time.",
                "Acceptable repair for my {device} {issue}. Met expectations but didn't exceed them. Fair pricing.",
                "Okay experience overall. My {device} {issue} is working but customer service could improve.",
                "Mediocre service. The {device} {issue} was eventually fixed but had some delays. Average quality.",
                "So-so experience. They handled my {device} {issue} adequately but the process wasn't smooth."
            ],
            2: [
                "Disappointing service. My {device} {issue} took way too long to fix. Not impressed with the communication.",
                "Below expectations. The {device} {issue} repair was slow and expensive. Would probably look elsewhere next time.",
                "Not great. My {device} {issue} works now but I had issues with the initial repair. Had to bring it back.",
                "Subpar experience. The {device} {issue} fix took longer than promised and customer service was lacking.",
                "Underwhelming service for my {device} {issue}. Overpriced and the repair quality could be better.",
                "Poor communication. My {device} {issue} repair took forever and nobody kept me updated. Frustrating experience.",
                "Not satisfied. The {device} {issue} problem persists somewhat even after the repair. May need to go elsewhere.",
                "Slow service. My {device} {issue} took much longer than estimated. The technician seemed inexperienced."
            ],
            1: [
                "Terrible experience! My {device} {issue} is still not fixed properly. Wasted my time and money. Avoid!",
                "Horrible service! They made my {device} {issue} worse. Unprofessional and incompetent. Do not recommend!",
                "Awful! My {device} {issue} repair failed after two days. Refused refund and was rude. Stay away!",
                "Worst repair shop! The {device} {issue} wasn't fixed and they charged me anyway. Extremely disappointed.",
                "Disaster! My {device} {issue} is worse than before. Poor workmanship and bad customer service. Never again!",
                "Completely useless. My {device} {issue} still broken after their 'repair'. Waste of money and time.",
                "Terrible! They damaged my {device} while trying to fix the {issue}. Unprofessional and careless staff.",
                "Nightmare experience. The {device} {issue} repair failed immediately. No accountability. Avoid at all costs!"
            ]
        }
        
        # Device types and issues
        self.device_issues = {
            "iPhone": ["screen replacement", "battery issue", "charging port problem", "water damage", "broken camera"],
            "Samsung phone": ["cracked screen", "battery replacement", "charging issue", "speaker problem", "power button"],
            "Android phone": ["screen repair", "battery issue", "software problem", "charging port", "camera issue"],
            "laptop": ["hard drive failure", "keyboard replacement", "screen issue", "overheating problem", "virus removal"],
            "desktop computer": ["network issue", "hardware upgrade", "blue screen error", "slow performance", "power supply"],
            "MacBook": ["battery replacement", "keyboard issue", "screen repair", "trackpad problem", "charging issue"],
            "iPad": ["cracked screen", "charging port", "battery replacement", "software update", "speaker issue"],
            "gaming PC": ["graphics card issue", "cooling problem", "RAM upgrade", "performance optimization", "driver issue"]
        }
    
    def generate_email(self, name: str) -> str:
        """Generate a realistic email from a name."""
        name_parts = name.lower().split()
        patterns = [
            f"{name_parts[0]}.{name_parts[1]}",
            f"{name_parts[0]}{name_parts[1]}",
            f"{name_parts[0][0]}{name_parts[1]}",
            f"{name_parts[0]}_{name_parts[1]}",
            f"{name_parts[0]}{random.randint(10, 99)}"
        ]
        return f"{random.choice(patterns)}@{random.choice(self.email_domains)}"
    
    def generate_review(self, rating: int) -> Tuple[str, str, str]:
        """Generate a realistic review with device and issue context."""
        device = random.choice(list(self.device_issues.keys()))
        issue = random.choice(self.device_issues[device])
        
        template = random.choice(self.review_templates[rating])
        
        # Some variation in technician mentions
        tech_names = ["", "John", "Sarah", "Mike", "the technician"]
        name = random.choice(tech_names)
        
        review = template.format(device=device, issue=issue, name=name)
        
        # Add occasional extra details
        if random.random() < 0.3:
            extras = [
                " The waiting area was comfortable.",
                " Pricing was transparent and fair.",
                " They explained the issue clearly.",
                " Got a warranty on the repair.",
                " Same-day service was appreciated.",
                " Clean and organized shop.",
                " Parking was convenient."
            ]
            review += random.choice(extras)
        
        return review, device, issue
    
    def seed_reviews(self, num_reviews: int = 50, days_back: int = 90):
        """Seed the database with realistic customer reviews."""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            print(f"Seeding {num_reviews} customer reviews...")
            print(f"Connecting to database: {self.db_config['database']}")
            
            # Rating distribution (realistic for a decent repair shop)
            # More 4-5 star reviews, fewer 1-2 star reviews
            rating_distribution = [5]*30 + [4]*25 + [3]*20 + [2]*15 + [1]*10
            
            success_count = 0
            
            for i in range(num_reviews):
                # Random customer
                name = random.choice(self.names)
                email = self.generate_email(name)
                
                # Random rating weighted by distribution
                rating = random.choice(rating_distribution)
                
                # Generate review
                review_text, device, issue = self.generate_review(rating)
                
                # Random timestamp within the past X days
                days_ago = random.randint(0, days_back)
                hours_ago = random.randint(0, 23)
                minutes_ago = random.randint(0, 59)
                timestamp = datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
                
                # Insert into database
                try:
                    cursor.execute("""
                        INSERT INTO comments 
                        (customer_name, email, rating, review_content, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (name, email, rating, review_text, timestamp, timestamp))
                    
                    success_count += 1
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Inserted {i + 1}/{num_reviews} reviews...")
                        
                except mysql.connector.Error as e:
                    print(f"  Error inserting review {i + 1}: {e}")
                    continue
            
            conn.commit()
            print(f"\n✓ Successfully seeded {success_count} customer reviews!")
            
            # Show statistics
            cursor.execute("SELECT rating, COUNT(*) as count FROM comments GROUP BY rating ORDER BY rating DESC")
            results = cursor.fetchall()
            
            print("\nRating Distribution:")
            for rating, count in results:
                print(f"  {rating} stars: {count} reviews")
            
            cursor.execute("SELECT AVG(rating) as avg_rating FROM comments")
            avg_rating = cursor.fetchone()[0]
            print(f"\nAverage Rating: {avg_rating:.2f} stars")
            
            cursor.close()
            conn.close()
            
        except mysql.connector.Error as e:
            print(f"Database connection error: {e}")
            print("Please check your database credentials and ensure MySQL is running on port 3307")
    
    def clear_reviews(self):
        """Clear all existing reviews (use with caution!)"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            response = input("Are you sure you want to DELETE all customer reviews? (yes/no): ")
            if response.lower() == 'yes':
                cursor.execute("DELETE FROM comments")
                conn.commit()
                print(f"✓ Deleted all customer reviews")
            else:
                print("Cancelled.")
            
            cursor.close()
            conn.close()
            
        except mysql.connector.Error as e:
            print(f"Database error: {e}")


def main():
    seeder = ReviewSeeder()
    
    print("=" * 70)
    print("BingleFixIt Customer Review Seeder")
    print("=" * 70)
    print("\nOptions:")
    print("1. Seed reviews (default: 50 reviews)")
    print("2. Seed reviews (custom amount)")
    print("3. Clear all customer reviews")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        seeder.seed_reviews(num_reviews=50)
    elif choice == "2":
        try:
            num = int(input("How many reviews to seed? "))
            if num > 0 and num <= 1000:
                seeder.seed_reviews(num_reviews=num)
            else:
                print("Please enter a number between 1 and 1000")
        except ValueError:
            print("Invalid number")
    elif choice == "3":
        seeder.clear_reviews()
    elif choice == "4":
        print("Exiting...")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()