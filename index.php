<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Mobile Repair Services</title>
        
        <!-- Pure CSS -->
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/purecss@3.0.0/build/pure-min.css">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/purecss@3.0.0/build/grids-responsive-min.css">
        
        <style>
            /* General Styles */
            body {
                line-height: 1.6;
                padding: 0;
                margin: 0;
            }

            /* Banner Section */
            .banner {
                background-color: #2c3e50;
                color: white;
                padding: 4em 2em;
                margin-bottom: 2em;
                width: 100%;
                min-height: 200px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            /* Description Box */
            .description-box {
                padding: 2em;
                margin-bottom: 2em;
                font-size: 1.1em;
                line-height: 1.6;
                max-width: 800px;
                margin-left: 2em;
            }

            /* Services Section */
            .services-section {
                padding: 2em;
                margin-bottom: 2em;
            }

            .service-boxes {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 2em;
                padding: 0 2em;
            }

            .service-box {
                text-align: center;
                padding: 1em;
                background: #f8f9fa;
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }

            .badge-icon {
                width: 100%;
                max-width: 400px;
                height: auto;
                margin-bottom: 1em;
            }

            /* Review Section */
            .review-section {
                padding: 2em;
                max-width: 800px;
                margin: 0;
            }
            
            .review-form {
                margin-bottom: 2em;
            }
            
            .form-group {
                margin-bottom: 1em;
            }
            
            .review-box {
                padding: 1em;
                margin-bottom: 1em;
                background: #f8f9fa;
                border-radius: 4px;
            }
            
            .success-message {
                background: #dff0d8;
                color: #3c763d;
                padding: 1em;
                margin: 1em 0;
                border-radius: 4px;
            }
            
            .error-message {
                background: #f2dede;
                color: #a94442;
                padding: 1em;
                margin: 1em 0;
                border-radius: 4px;
            }

            .pure-form {
                text-align: left;
            }

            .pure-form label {
                text-align: left;
                display: block;
            }

            /* Responsive adjustments */
            @media screen and (max-width: 768px) {
                .service-boxes {
                    grid-template-columns: 1fr;
                    padding: 0 1em;
                }
                
                .description-box {
                    margin: 1em;
                }

                .banner {
                    padding: 2em 1em;
                }
            }
        </style>
    </head>

    <body>
        <header class="banner">
            <h1>Mobile Repair Services</h1>
        </header>

        <div class="description-box">
            Lorem ipsum dolor sit amet, 
            consectetur adipiscing elit. 
            Aliquam cursus lectus non tellus venenatis, vel bibendum nisi faucibus. 
            Proin vehicula dignissim magna a posuere. Donec tempus ex id arcu tempor pulvinar. 
            Phasellus mattis ullamcorper elit semper viverra. 
        </div>

        <div class="services-section">
            <div class="service-boxes">
                <div class="service-box">
                    <img src="https://placehold.co/400x400" alt="Service 1 Badge" class="badge-icon">
                    <p>Service 1 Description</p>
                </div>

                <div class="service-box">
                    <img src="https://placehold.co/400x400" alt="Service 2 Badge" class="badge-icon">
                    <p>Service 2 Description</p>
                </div>

                <div class="service-box">
                    <img src="https://placehold.co/400x400" alt="Service 3 Badge" class="badge-icon">
                    <p>Service 3 Description</p>
                </div>
            </div>
        </div>

        <!-- Rest of the review section code remains the same -->
        <div class="review-section">
            <h1 class="pure-u-1">Leave a Review</h1>
            
            <form class="pure-form pure-form-stacked review-form" action="" method="POST">
                <fieldset>
                    <div class="pure-g">
                        <div class="pure-u-1 pure-u-md-1-2 form-group">
                            <label for="name">Name:</label>
                            <input id="name" type="text" name="Name" class="pure-input-1" required>
                        </div>

                        <div class="pure-u-1 pure-u-md-1-2 form-group">
                            <label for="email">Email:</label>
                            <input id="email" type="email" name="Email" class="pure-input-1" required>
                        </div>

                        <div class="pure-u-1 pure-u-md-1-2 form-group">
                            <label for="rating">Rating (out of 5):</label>
                            <input id="rating" type="number" name="Rating" min="1" max="5" class="pure-input-1" required>
                        </div>

                        <div class="pure-u-1 form-group">
                            <label for="comment">Comment:</label>
                            <textarea id="comment" name="Comment" class="pure-input-1" rows="4" required></textarea>
                        </div>

                        <div class="pure-u-1">
                            <button type="submit" name="Submit" class="pure-button pure-button-primary">Submit Review</button>
                        </div>
                    </div>
                </fieldset>
            </form>

            <?php
            if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST["Submit"])) {
                $host = "localhost";
                $db_name = "mobile_repair";
                $username = "root";
                $password = "";

                try {
                    $conn = new mysqli($host, $username, $password, $db_name);

                    if ($conn->connect_error) {
                        throw new Exception("Connection failed: " . $conn->connect_error);
                    }

                    $customer_name = $_POST["Name"];
                    $email = $_POST["Email"];
                    $rating = $_POST["Rating"];
                    $review_content = $_POST["Comment"];

                    $stmt = $conn->prepare("INSERT INTO reviews (customer_name, email, rating, review_content, created_at, updated_at) VALUES (?, ?, ?, ?, NOW(), NOW())");
                    $stmt->bind_param("ssis", $customer_name, $email, $rating, $review_content);

                    if ($stmt->execute()) {
                        echo "<div class='success-message'>Your review has been submitted successfully!</div>";
                    } else {
                        throw new Exception("Error submitting review: " . $stmt->error);
                    }

                    $stmt->close();

                    // Display all reviews
                    $sql = "SELECT customer_name, email, rating, review_content, created_at FROM reviews ORDER BY created_at DESC";
                    $result = $conn->query($sql);

                    if ($result->num_rows > 0) {
                        echo "<h2 class='pure-u-1'>Customer Reviews</h2>";
                        
                        while ($row = $result->fetch_assoc()) {
                            echo "<div class='review-box pure-g'>
                                <div class='pure-u-1 pure-u-md-1-2'>
                                    <p><strong>Name:</strong> " . htmlspecialchars($row["customer_name"]) . "</p>
                                    <p><strong>Email:</strong> " . htmlspecialchars($row["email"]) . "</p>
                                </div>
                                <div class='pure-u-1 pure-u-md-1-2'>
                                    <p><strong>Rating:</strong> " . htmlspecialchars($row["rating"]) . "/5</p>
                                    <p><strong>Date:</strong> " . date("Y/m/d h:i A", strtotime($row["created_at"])) . "</p>
                                </div>
                                <div class='pure-u-1'>
                                    <p><strong>Review:</strong> " . htmlspecialchars($row["review_content"]) . "</p>
                                </div>
                            </div>";
                        }
                    } else {
                        echo "<p class='pure-u-1'>No reviews yet!</p>";
                    }

                    $conn->close();
                } catch (Exception $e) {
                    echo "<div class='error-message'>Error: " . htmlspecialchars($e->getMessage()) . "</div>";
                }
            }
            ?>
        </div>
    </body>
</html>