// 21BAI1225 - Swarab Raul

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class nav : MonoBehaviour
{
    public UnityEngine.AI.NavMeshAgent agent;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if(Input.GetMouseButtonDown(1)) {
            Ray movePosition = Camera.main.ScreenPointToRay(Input.mousePosition);

            if(Physics.Raycast(movePosition, out var hitInfo)) {
                agent.SetDestination(hitInfo.point);
            }
        }
    }
}
