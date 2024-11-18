// 21BAI1225 - Swarab Raul
// BCSE416P Game Programming - Lab 3 (Scripts In Unity)

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SphereScript : MonoBehaviour
{
    // Public variable to control the speed of the sphere's movement
    public float speed = 5.0f;

    // Awake is called when the script instance is being loaded
    void Awake()
    {
        Debug.Log("Awake - 21BAI1225(Swarab Raul)");
    }

    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("\nStart - 21BAI1225(Swarab Raul)");
        print("\nMy First Script - 21BAI1225(Swarab Raul)");
    }

    // Update is called once per frame
    void Update()
    {
        Debug.Log("Update Time: " + Time.deltaTime + " - 21BAI1225(Swarab Raul)");

        MovePlayer();
    }

    // FixedUpdate is called at fixed intervals
    void FixedUpdate()
    {
        Debug.Log("FixedUpdate Time: " + Time.deltaTime + " - 21BAI1225(Swarab Raul)");
    }

    // Function to move the player based on input from the keyboard (WASD or arrow keys, spacebar, and shift key)
    void MovePlayer()
    {
        float moveHorizontal = Input.GetAxis("Horizontal");
        float moveVertical = Input.GetAxis("Vertical");

        float moveUpDown = 0.0f;

        if (Input.GetKey(KeyCode.Space))
        {
            moveUpDown = 1.0f;
        }

        if (Input.GetKey(KeyCode.LeftShift))
        {
            moveUpDown = -1.0f;
        }

        Vector3 movement = new Vector3(moveHorizontal, moveUpDown, moveVertical);
        transform.Translate(movement * speed * Time.deltaTime, Space.World);
    }

    // OnCollisionEnter is called when this collider/rigidbody has begun
    void OnCollisionEnter(Collision collision)
    {
        // Check if the object we collided with has the tag "Wall"
        if (collision.gameObject.CompareTag("Wall"))
        {
            Debug.Log("Collided with Wall - Object Destroyed - 21BAI1225(Swarab Raul)");
            
            // Destroy the current game object (the sphere)
            Destroy(gameObject);
        }
    }
}
