using UnityEngine;
using UnityEngine.SceneManagement;

public class StarTrigger : MonoBehaviour
{
    private void OnTriggerEnter2D(Collider2D collision)
    {
        // Check if the object colliding is the character
        if (collision.CompareTag("Player"))
        {
            // Load the second scene
            SceneManager.LoadScene("scene2");
        }
    }
}
